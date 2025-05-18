from pyomo.environ import *
from pyomo.opt import SolverFactory
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_bus import BUS_TYPE, REF, BUS_I

class CLMP:
    def __init__(self, net, solver, verbose=False):
        self.net = net        
        self.solver = solver
        self.verbose = verbose
        self.model = ConcreteModel()
        self.create_model()
        
    def create_model(self):
        
        pp.rundcpp(self.net)  
        ppc = pp.converter.to_ppc(self.net)

        # 提取基准容量、节点数据和支路数据
        baseMVA = ppc['baseMVA']
        gen = ppc['gen']
        gen_idx = range(gen.shape[0])
        bus = ppc['bus']
        bus_idx = range(bus.shape[0])
        branch = ppc['branch']
        branch_idx = range(branch.shape[0])
        ref_bus = np.where(bus[:, BUS_TYPE] == REF)[0][0]
        
        PTDF_matrix = makePTDF(baseMVA, bus, branch, slack=ref_bus,
                result_side=0, using_sparse_solver=True, branch_id=None, reduced=False)
        BusGen_matrix = np.zeros((len(bus_idx), len(gen_idx)))
        for i in bus_idx:
            for j in gen_idx:
                if gen[j, 0] == i:
                    BusGen_matrix[i, j] = 1
        
        def PTDF(model, i, j):
            return PTDF_matrix[i, j]

        def BusGen(model, i, j):
            return BusGen_matrix[i, j]
    
        Cost = self.net.poly_cost['cp1_eur_per_mw']    
        Emission = pd.DataFrame([10]*len(gen_idx), index=gen_idx, columns=['emission_factor'])
    
        # 创建模型参数            
        self.model.Pg_max = Param(gen_idx, initialize=gen[:, 8]) #'PMAX'
        self.model.Pg_min = Param(gen_idx, initialize=gen[:, 9]) #'PMIN'
        self.model.Pf_max = Param(branch_idx, initialize=branch[:, 5]) #'RATE_A'
        self.model.Pd = Param(bus_idx, initialize=bus[:, 2]) #PD
        self.model.C = Param(gen_idx, initialize=Cost)
        self.model.E = Param(gen_idx, initialize=Emission) 
        self.model.BusGen = Param(bus_idx, gen_idx, initialize=BusGen)
        self.model.PTDF = Param(branch_idx, bus_idx, initialize=PTDF) 
        
        # 创建模型变量
        Pg = gen[:, 1] 
        Pf = np.concatenate((self.net.res_line.p_from_mw, self.net.res_trafo.p_hv_mw))
        Pf_pos = np.clip(Pf, 0, 100000)
        Pf_neg = - np.clip(Pf, -100000, 0)     
        Pg_bus = np.matmul(BusGen_matrix, Pg)
        Dir = np.sign(Pf)
        Dir[Dir<0] = 0
        self.model.Pg = Var(gen_idx, domain=Reals, initialize=Pg) #'PG'
        self.model.Pg_bus = Var(bus_idx, domain=Reals, initialize=Pg_bus)
        self.model.Pf = Var(branch_idx, domain=Reals, initialize=Pf) #'PF'-'PT'
        self.model.Pf_pos = Var(branch_idx, domain=Reals, initialize=Pf_pos) #'PF'
        self.model.Pf_neg = Var(branch_idx, domain=Reals, initialize=Pf_neg) #'PT'
        self.model.CI = Var(bus_idx, domain=Reals, initialize=[10]*len(bus_idx))
        self.model.Dir = Var(branch_idx, domain=Binary, initialize=Dir)
     
        # 创建模型约束
        def power_balance(model, i):
            return model.Pg_bus[i] +  sum(model.Pf[k] for k in branch_idx if branch[k, 1] == i) - \
            sum(model.Pf[k] for k in branch_idx if branch[k, 0] == i) == model.Pd[i]    # F_BUS(0), T_BUS(1)
        def Pg_bus_equality(model, i):
            return model.Pg_bus[i] == sum(model.Pg[j] * model.BusGen[i, j] for j in gen_idx)
        def Pf_equality(model, k):
            return model.Pf[k] == sum((model.Pg_bus[i] - model.Pd[i]) * model.PTDF[k, i] for i in bus_idx)
        def Pg_max_inequality(model, j):
            return model.Pg[j] <= model.Pg_max[j]
        def Pg_min_inequality(model, j):
            return model.Pg[j] >= model.Pg_min[j]
        def Pf_decompose(model, k):
            return model.Pf[k] == model.Pf_pos[k] - model.Pf_neg[k] 
        def Pf_pos_upper(model, k):
            return model.Pf_pos[k] <= self.model.Dir[k] * model.Pf_max[k]
        def Pf_pos_lower(model, k):
            return 0 <= model.Pf_pos[k]
        def Pf_neg_upper(model,k):
            return model.Pf_neg[k] <= (1 - self.model.Dir[k]) * model.Pf_max[k]
        def Pf_neg_lower(model, k):
            return 0 <= model.Pf_neg[k] 
        def Carbon_equation(model, i):
            return model.CI[i] * (sum(model.Pg[j] * model.BusGen[i, j] for j in gen_idx) + \
                                  sum(model.Pf_pos[k] for k in branch_idx if branch[k, 1] == i) + \
                                  sum(model.Pf_neg[k] for k in branch_idx if branch[k, 0] == i)) \
                == sum(model.Pg[j] * model.BusGen[i, j] * model.E[j] for j in gen_idx) + \
                   sum(model.Pf_pos[k] * model.CI[branch[k, 0]] for k in branch_idx if branch[k, 1] == i) + \
                   sum(model.Pf_neg[k] * model.CI[branch[k, 1]] for k in branch_idx if branch[k, 0] == i)
        def Carbon_cap(model):
            return sum(model.CI[i] * model.Pd[i] for i in bus_idx) <= 10000000
        
        self.model.power_balance = Constraint(bus_idx, rule=power_balance)
        self.model.Pg_bus_equality = Constraint(bus_idx, rule=Pg_bus_equality)
        self.model.Pf_equality = Constraint(branch_idx, rule=Pf_equality)
        self.model.Pg_max_inequality = Constraint(gen_idx, rule=Pg_max_inequality)
        self.model.Pg_min_inequality = Constraint(gen_idx, rule=Pg_min_inequality)
        self.model.Pf_decompose = Constraint(branch_idx, rule=Pf_decompose)
        self.model.Pf_pos_upper = Constraint(branch_idx, rule=Pf_pos_upper)
        self.model.Pf_pos_lower = Constraint(branch_idx, rule=Pf_pos_lower)
        self.model.Pf_neg_upper = Constraint(branch_idx, rule=Pf_neg_upper)
        self.model.Pf_neg_lower = Constraint(branch_idx, rule=Pf_neg_lower)
        self.model.Carbon_equation = Constraint(bus_idx, rule=Carbon_equation)
        self.model.Carbon_cap = Constraint(rule=Carbon_cap)
            
        # 创建模型目标
        self.model.objective = Objective(
            expr=sum(self.model.Pg[j] * self.model.C[j] for j in gen_idx), sense=minimize)

    def solve(self):
        opt = SolverFactory(self.solver)
        results = opt.solve(self.model, tee=self.verbose)
        print(results)
        return results
    

if __name__ == "__main__":
    
    # 创建一个示例网络
    net = pn.case118()
    # 创建一个CLMP模型
    clmp = CLMP(net, solver='ipopt', verbose=True)
    clmp.solve()