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
from pandapower.converter.pypower.to_ppc import _pd2ppc

class CLMP:
    def __init__(self, net, solver='gurobi', verbose=False):
        self.net = net        
        self.solver = solver
        self.verbose = verbose
        self.model = ConcreteModel()
        self.create_model()
        
    def create_model(self):
        
        pp.rundcpp(self.net)  
        ppc = _pd2ppc(self.net)

        # 提取基准容量、节点数据和支路数据
        baseMVA = ppc[0]['baseMVA']
        gen = ppc[0]['gen']
        gen_idx = range(gen.shape[0])
        bus = ppc[0]['bus']
        bus_idx = range(bus.shape[0])
        branch = ppc[0]['branch']
        branch_idx = range(branch.shape[0])
        ref_bus = np.where(bus[:, BUS_TYPE] == REF)[0][0]
        
        PTDF_matrix = makePTDF(baseMVA, bus, branch, slack=ref_bus,
                result_side=0, using_sparse_solver=True, branch_id=None, reduced=False)
        def PTDF(model, i, j):
            return PTDF_matrix[i, j]

        def BusGen(model, i, j):
            if gen[j, 0] == i:
                return 1
            else: 
                return 0
            
        C = self.net.poly_cost['cp1_eur_per_mw']    
        
        # 创建模型参数            
        self.model.Pg_max = Param(gen_idx, initialize=gen[:, 8]) #'PMAX'
        self.model.Pg_min = Param(gen_idx, initialize=gen[:, 9]) #'PMIN'
        self.model.Pf_max = Param(branch_idx, initialize=branch[:, 5].real) #'RATE_A'
        self.model.Pd = Param(bus_idx, initialize=bus[:, 2]) #PD
        self.model.C = Param(gen_idx, initialize=C)
        self.model.GB = Param(bus_idx, gen_idx, initialize=BusGen)
        self.model.PTDF = Param(branch_idx, bus_idx, initialize=PTDF) 
        
        # 创建模型变量
        self.model.Pg = Var(gen_idx, domain=NonNegativeReals)
        self.model.Pg_bus = Var(bus_idx, domain=NonNegativeReals)
        self.model.Pf = Var(branch_idx, domain=Reals)
     
        # 创建模型约束
        def power_balance(model, i):
            return model.Pg_bus[i] +  sum(model.Pf[j] for j in branch_idx if branch[j, 0] == i) - \
            sum(model.Pf[j] for j in branch_idx if branch[j, 1] == i) == model.Pd[i]    # T_BUS(0), F_BUS(1)
        def Pg_bus_equality(model, i):
            return model.Pg_bus[i] == sum(model.Pg[j] * model.GB[i, j] for j in gen_idx)
        def Pf_equality(model, i):
            return model.Pf[i] == sum((model.Pg_bus[j] - model.Pd[j])* model.PTDF[i, j] for j in bus_idx)
        def Pg_max_inequality(model, i):
            return model.Pg[i] <= model.Pg_max[i]
        def Pg_min_inequality(model, i):
            return model.Pg[i] >= model.Pg_min[i]
        def Pf_max_inequality(model, i):
            return model.Pf[i] <= model.Pf_max[i]
        def Pf_min_inequality(model, i):
            return model.Pf[i] >= -model.Pf_max[i]

        self.model.power_balance = Constraint(bus_idx, rule=power_balance)
        self.model.Pg_bus_equality = Constraint(bus_idx, rule=Pg_bus_equality)
        self.model.Pf_equality = Constraint(branch_idx, rule=Pf_equality)
        self.model.Pg_max_inequality = Constraint(gen_idx, rule=Pg_max_inequality)
        self.model.Pg_min_inequality = Constraint(gen_idx, rule=Pg_min_inequality)
        self.model.Pf_max_inequality= Constraint(branch_idx, rule=Pf_max_inequality)
        self.model.Pf_min_inequality= Constraint(branch_idx, rule=Pf_min_inequality)
        
        # 创建模型目标
        self.model.objective = Objective(
            expr=sum(self.model.Pg[i] * self.model.C[i] for i in gen_idx), sense=minimize)

    def solve(self):
        opt = SolverFactory(self.solver)
        results = opt.solve(self.model, tee=self.verbose)
        print(results)
        return results
    

if __name__ == "__main__":
    
    # 创建一个示例网络
    net = pn.case118()
    # 创建一个CLMP模型
    clmp = CLMP(net, solver='gurobi', verbose=True)
    clmp.solve()