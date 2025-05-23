import math
import random
import numpy as np

from datetime import datetime
from pyomo.environ import (ConcreteModel, Set, Param, Var, Constraint, ConstraintList, Objective, Suffix, SolverFactory,
                           Reals, NonNegativeReals, minimize)

import pandapower as pp
import pandapower.networks as pn
import pandapower.converter as pc
from pandapower.pypower.makeBdc import makeBdc
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pypower.idx_gen import PG, GEN_BUS, PMAX, PMIN
# from pandapower.pypower.idx_cost import NCOST, COST
from pandapower.pypower.idx_brch import PF, PT, QF, QT, RATE_A, F_BUS, T_BUS
from pandapower.pypower.idx_bus import PD

from pypower.api import runpf, ppoption

ipopt_path = r'C:\Users\RichriD\Downloads\Compressed\Ipopt-3.14.17-win64-msvs2022-md\bin\ipopt.exe'


# structre_path = 'model_structure-' + run_id + '.txt'
# variable_path = 'model_variable-' + run_id + '.txt'

def find_all_branches(branch, from_bus, to_bus):
    """
    查找两个节点之间的支路, 返回值为支路在 branch 中的索引列表
    需要分成两个列表, 分别对应与给出参考方向相同的和相反的
    :param branch:
    :param from_bus:
    :param to_bus:
    :return:
    """
    for_brch = [
        i for i, br in enumerate(branch)
        if (int(br[F_BUS]) == from_bus and int(br[T_BUS]) == to_bus)
    ]
    back_brch = [
        i for i, br in enumerate(branch)
        if (int(br[F_BUS]) == to_bus and int(br[T_BUS]) == from_bus)
    ]
    return for_brch, back_brch


class CLMP:
    def __init__(self, net, e_g, solver_name, verbose, path, solver_executable=ipopt_path):
        self.net = net
        self.e_g = e_g
        self.solver_name = solver_name
        self.verbose = verbose
        self.path = path
        self.model = None
        self.solver_executable = solver_executable
        # self.T = None
        self.bus_adj = None
        self.adj_set = None

        # 将 pandapower 转为 ppc, 同时重新调用潮流求解算法
        pp.runpp(self.net)
        ppc = pc.to_ppc(self.net)
        ppopt = ppoption(PF_ALG=2, VERBOSE=0, OUT_ALL=0)
        self.res, _ = runpf(ppc, ppopt)
        self.baseMVA, self.bus, self.gen, self.branch = (
            self.res["baseMVA"], self.res["bus"], self.res["gen"], self.res["branch"])

        # 导出节点导纳矩阵 B
        self.Bbus, _, _, _, _ = makeBdc(self.bus, self.branch)
        self.B = dict(self.Bbus.todok().items())

        self.T = makePTDF(self.baseMVA, self.bus, self.branch)

        # 按照原支路顺序导出的支路始末节点 list
        from_bus = self.branch[:, 0].astype(int)
        to_bus = self.branch[:, 1].astype(int)
        self.line_list = list(zip(from_bus, to_bus))

        # 节点数, 发电机数量, 发电机节点, 以及节点类型
        self.bus_num = self.bus.shape[0]
        self.gen_num = self.gen.shape[0]
        self.gen_bus = self.gen[:, GEN_BUS].astype(int)
        self.type_list = self.bus[:, 1]

        # cost_dic 为以字典格式存储的发电成本矩阵, 表示对应节点 (row) 上发电机的二次成本函数
        bus_gen_mtrx = np.zeros((self.bus_num, 3))
        bus_gen_mtrx[self.gen_bus, :] = self.res['gencost'][:, -3:]
        self.cost_dic = {(i, j): value
                         for i, row in enumerate(bus_gen_mtrx)
                         for j, value in enumerate(row)}

        # 直接初始化发电机的 CEI 是暂行办法, 需要判断是否属于集合 \mathcal{J}
        self.ei = np.zeros(self.bus_num) * min(self.e_g)
        self.ei[self.gen_bus] = self.e_g

        self.create_model()
        self.adj_matrix()  # 邻接矩阵以及邻接列表 (用于索引)
        # self.cal_T()

    def adj_matrix(self):
        # 节点邻接矩阵可以直接通过 B 得到 (去除对角元)
        Bbus_array = self.Bbus.toarray() if hasattr(self.Bbus, "toarray") else np.array(self.Bbus)
        # 移除对角元（设为 0）
        np.fill_diagonal(Bbus_array, 0)
        # 所有非零元素设为 1
        self.bus_adj = (Bbus_array != 0).astype(int)
        self.adj_set = []
        for i in range(self.bus_num):
            self.adj_set.append(set(np.where(self.bus_adj[i, :] == 1)[0]))

    def create_model(self):
        self.model = ConcreteModel()

        # --- 前置 ---
        pg_max_list = np.zeros(self.bus_num)
        pg_min_list = np.zeros(self.bus_num)
        pg_list = np.zeros(self.bus_num)
        pg_max_list[self.gen_bus] = self.gen[:, PMAX] / self.baseMVA  # 发电机出力上限（p.u.）
        pg_min_list[self.gen_bus] = self.gen[:, PMIN] / self.baseMVA  # 发电机出力下限（p.u.）
        pg_list[self.gen_bus] = self.gen[:, PG] / self.baseMVA  #

        # --- sets ---
        self.model.buses = Set(initialize=self.bus[:, 0].astype(int))
        self.model.lines = Set(initialize=range(len(self.line_list)))
        self.model.cost_dims = Set(initialize=range(0, 3))

        # --- params ---
        self.model.PD = Param(self.model.buses, initialize=self.bus[:, PD] / self.baseMVA, mutable=True)  # 可以改变部分负荷

        # 发电机出力上下限
        self.model.PG_MAX = Param(self.model.buses, initialize=pg_max_list, mutable=False)
        self.model.PG_MIN = Param(self.model.buses, initialize=pg_min_list, mutable=False)

        self.model.B = Param(self.model.buses, self.model.buses, initialize=self.B, default=0.0,
                             mutable=False)  # 节点导纳矩阵
        self.model.C = Param(self.model.buses, self.model.cost_dims, initialize=self.cost_dic, default=0.0,
                             mutable=False)  # 发电成本

        # 支路潮流限额（双向限额相同）
        PLMAX = self.branch[:, RATE_A] / self.baseMVA
        self.model.PL_MAX = Param(self.model.lines, initialize=PLMAX, default=0.0, mutable=False)

        # FIXME 暂时方法
        Emis_cap = np.ones(self.bus_num) * 0.8
        self.model.Emis_cap = Param(self.model.buses, initialize=Emis_cap, mutable=False)

        # FIXME 只对 IEEE 9-bus 生效
        self.model.w_g = Param(self.model.buses, initialize=self.ei, default=0.0)

        # --- vars ---
        self.model.PG = Var(self.model.buses, domain=Reals, initialize=pg_list)
        self.model.theta = Var(self.model.buses, bounds=(-math.pi / 8, math.pi / 8), initialize=0)
        self.model.w = Var(self.model.buses, bounds=(0, 0.8), initialize=self.ei)

        # flow vars
        # self.model.p_out = Var(self.model.buses, self.model.buses, domain=NonNegativeReals, initialize=0)
        # self.model.p_in = Var(self.model.buses, self.model.buses, domain=NonNegativeReals, initialize=0)
        self.model.p_out = Var(self.model.buses, self.model.buses, domain=Reals, initialize=0)
        self.model.p_in = Var(self.model.buses, self.model.buses, domain=Reals, initialize=0)
        for i in self.model.buses:  # 自回路置零
            self.model.p_out[i, i].fix(0)
            self.model.p_in[i, i].fix(0)
        for i in self.model.buses:  # 将不存在的支路始终置零 (根据节点导纳矩阵判断)
            for j in set(np.where(self.Bbus.toarray()[i, :] == 0)[0]):
                self.model.p_out[i, j].fix(0)
                self.model.p_in[i, j].fix(0)

        # 根据节点类型定义发电机限额, 以及固定平衡节点的电压相角
        for i, bus_type in enumerate(self.type_list):
            if bus_type in {2, 3}:
                # self.model.PG[i].lb = pg_min_list[i]
                # self.model.PG[i].ub = pg_max_list[i]
                self.model.w[i].fix(self.ei[i])
                if self.type_list[i] == 3:
                    self.model.theta[i].fix(0)
            else:
                self.model.PG[i].fix(0)

        # --- constraints ---
        # 0. 双向流方向松弛 & 大于 0 的约束
        self.model.branch_flow_dir = Constraint(self.model.buses, self.model.buses,
                                                rule=lambda m, i, j: m.p_out[i, j] * m.p_in[i, j] <= 0)
        self.model.for_flow_nonneg = Constraint(self.model.buses, self.model.buses,
                                                rule=lambda m, i, j: -m.p_out[i, j] <= 0)
        self.model.back_flow_nonneg = Constraint(self.model.buses, self.model.buses,
                                                 rule=lambda m, i, j: -m.p_in[i, j] <= 0)

        # NOTE pyomo 中定义的集合在进行索引的时候是从 1 开始的, 也就是说最好不要交叉混用索引.
        # 1. 功率平衡 (DCPF)约束, 按照节点顺序索引 -> \lambda_1
        # p^g_i - p^d_i = \sum_{j \in \mathcal{N}_i} (p^{out}_{ij} - p^{in}_{ij})
        # 结合上述等式约束"self.model.forward_branch_flow"以及"self.model.backward_branch_flow", 
        # 该约束的作用与原节点功率平衡约束相同
        self.model.power_balance_constraints = ConstraintList()
        for i in self.model.buses:
            # real_power_out = self.model.p_out[i] - self.model.p_in[i]
            real_power_out = sum(
                self.model.p_out[i, j] - self.model.p_in[i, j]
                for j in self.model.buses  # 变量定义的时候已经将不存在的支路始终置零
            )
            if self.type_list[i] in {2, 3}:  # 根据节点类型添加
                # self.model.power_balance_constraints.add(expr=self.model.PG[i] - self.model.PD[i] - real_power_out == 0)
                self.model.power_balance_constraints.add(expr=-self.model.PG[i] + self.model.PD[i] + real_power_out == 0)
            else:
                # self.model.power_balance_constraints.add(expr=- self.model.PD[i] - real_power_out == 0)
                self.model.power_balance_constraints.add(expr=self.model.PD[i] + real_power_out == 0)
        # 2. 双向潮流等效约束, 约束数量理论上为 2l 条, 分别写在两个list中, -> lambda_2
        self.model.forward_flow_eq = ConstraintList()
        self.model.backward_flow_eq = ConstraintList()
        # 3. 同时还有输电容量约束, 利用同一个 for 循环添加完成 -> mu^+ & mu^-
        self.model.branch_capacity = ConstraintList()
        for l in self.model.lines:
            fbus = self.branch[l, F_BUS]  # i
            tbus = self.branch[l, T_BUS]  # j
            B_ft = -self.model.B[fbus, tbus]
            pf_ij = B_ft * (self.model.theta[fbus] - self.model.theta[tbus])

            self.model.forward_flow_eq.add(
                expr=self.model.p_out[fbus, tbus] - self.model.p_in[fbus, tbus] == pf_ij
            )
            self.model.backward_flow_eq.add(
                expr=self.model.p_out[tbus, fbus] - self.model.p_in[tbus, fbus] == -pf_ij
            )

            self.model.branch_capacity.add(
                expr=-self.model.PL_MAX[l] - (self.model.p_out[fbus, tbus] - self.model.p_in[fbus, tbus]) <= 0
            )
            self.model.branch_capacity.add(
                expr=(self.model.p_out[fbus, tbus] - self.model.p_in[fbus, tbus]) - self.model.PL_MAX[l] <= 0
            )

        # 4. 发电功率约束 -> \eta
        self.model.generator_limits = ConstraintList()
        for i in self.model.buses:
            if self.type_list[i] in {2, 3}:
                self.model.generator_limits.add(expr=self.model.PG_MIN[i] - self.model.PG[i] <= 0)
                self.model.generator_limits.add(expr=self.model.PG[i] - self.model.PG_MAX[i] <= 0)

        # 5. 碳流方程 -> \epsilon
        self.model.carbon_emission_flow = ConstraintList()
        for i in self.model.buses:
            # 功率注入
            p_inj = self.model.PG[i] + sum(self.model.p_in[i, j] for j in self.model.buses)
            # 碳流注入 = 发电机注入                  +  支路注入
            emis_inj = self.model.w_g[i] * self.model.PG[i] + sum(
                self.model.p_in[i, j] * self.model.w[j] for j in self.model.buses)  # 变量约束中指明了不存在的支路与自回路对应的功率为 0
            self.model.carbon_emission_flow.add(expr=p_inj * self.model.w[i] == emis_inj)

        # 6. 碳排放上限 -> \tau
        self.model.carbon_cap = Constraint(self.model.buses,
                                           rule=lambda m, i: m.w[i] * m.PD[i] <= m.Emis_cap[i])

        # --- objective ---
        self.model.obj = Objective(
            expr=sum((self.model.PG[i] * self.baseMVA) ** 2 * self.model.C[i, 0]
                     + (self.model.PG[i] * self.baseMVA) * self.model.C[i, 1]
                     + self.model.C[i, 2]
                     for i in self.model.buses),
            sense=minimize)
        # suffix for duals
        self.model.dual = Suffix(direction=Suffix.IMPORT)
        # self.model = m

    # # —— 修改接口示例 ——
    # def update_loads(self, new_PD: dict):
    #     """new_PD: {bus_index: PD_pu, ...}"""
    #     for i, val in new_PD.items():
    #         self.model.PD[i] = val
    #
    # def update_emission_caps(self, new_caps: dict):
    #     for i, cap in new_caps.items():
    #         self.model.Emis_cap[i] = cap
    #
    # def update_cost_coeffs(self, new_C: dict):
    #     """new_C: {(i,k): coeff, ...}"""
    #     for key, val in new_C.items():
    #         self.model.C[key] = val

    # ——————————————————
    def opt_solve(self, tol=1e-6, acp_tol=1e-5):
        solver = SolverFactory(self.solver_name, executable=self.solver_executable)
        solver.options['tol'] = tol  # 允许更大的误差
        solver.options['acceptable_tol'] = acp_tol
        return solver.solve(self.model, tee=self.verbose)

    def export_results(self):
        # 打印结构到文件
        with open(self.path[0], 'w') as f:
            self.model.pprint(ostream=f)
        with open(self.path[1], 'w') as f:
            self.model.display(ostream=f)
        with open(self.path[2], 'w') as f:
            for c in self.model.component_objects(Constraint, active=True):
                f.write(f'Constraint block: {c.name}\n')
                for idx in getattr(self.model, c.name):
                    f.write(str(getattr(self.model, c.name)[idx].expr) + '\n')
                f.write('\n')


def main():
    # 路径改为你的 ipopt 可执行文件路径
    # ipopt_path = r'C:\Users\RichriD\Downloads\Compressed\Ipopt-3.14.17-win64-msvs2022-md\bin\ipopt.exe'
    case = 'case9'
    net = pn.case9()
    e_g = [0.3, 0.5, 0.8]

    run_id = case + '-' + datetime.now().strftime("%Y%m%d") + '-' + str(random.randint(0, 9999))
    structre_path = 'res/model_structure-' + run_id + '.txt'
    variable_path = 'res/model_variable-' + run_id + '.txt'
    constraints_path = 'res/constraints-' + run_id + '.txt'
    path_list = [structre_path, variable_path, constraints_path]

    clmp = CLMP(net, e_g, solver_name='ipopt', verbose=True, path=path_list)
    clmp.create_model()
    _ = clmp.opt_solve()
    clmp.export_results()


if __name__ == '__main__':
    main()
