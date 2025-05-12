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
    
        Cost = self.net.poly_cost['cp1_eur_per_mw']    
        Emission = pd.DataFrame([10]*len(gen_idx), index=gen_idx, columns=['emission_factor'])
    
        # 创建模型参数            
        self.model.Pg_max = Param(gen_idx, initialize=gen[:, 8]) #'PMAX'
        self.model.Pg_min = Param(gen_idx, initialize=gen[:, 9]) #'PMIN'
        self.model.Pf_max = Param(branch_idx, initialize=branch[:, 5].real) #'RATE_A'
        self.model.Pd = Param(bus_idx, initialize=bus[:, 2]) #PD
        self.model.C = Param(gen_idx, initialize=Cost)
        self.model.E = Param(gen_idx, initialize=Emission) 
        self.model.BusGen = Param(bus_idx, gen_idx, initialize=BusGen)
        self.model.PTDF = Param(branch_idx, bus_idx, initialize=PTDF) 
        self.model.M = Param(initialize=1e5)
        
        # 创建模型变量
        self.model.Pg = Var(gen_idx, domain=Reals, initialize=gen[:, 1]) #'PG'
        
        self.model.Pg_bus = Var(bus_idx, domain=Reals)
        self.model.Pf = Var(branch_idx, domain=Reals, initialize=branch[:, 13].real - branch[:, 15].real) #'PF'-'PT'
        self.model.Pf_pos = Var(branch_idx, domain=Reals, initialize=branch[:, 13].real) #'PF'
        self.model.Pf_neg = Var(branch_idx, domain=Reals, initialize=branch[:, 15].real) #'PT'
        self.model.CI = Var(bus_idx, domain=Reals)
        self.model.Dir = Var(branch_idx, domain=Binary)
     
        # 创建模型约束
        def power_balance(model, i):
            return model.Pg_bus[i] +  sum(model.Pf[i] for j in branch_idx if branch[j, 1] == i) - \
            sum(model.Pf[i] for j in branch_idx if branch[j, 0] == i) == model.Pd[i]    # F_BUS(0), T_BUS(1)
        def Pg_bus_equality(model, i):
            return model.Pg_bus[i] == sum(model.Pg[j] * model.BusGen[i, j] for j in gen_idx)
        def Pf_equality(model, i):
            return model.Pf[i] == sum((model.Pg_bus[j] - model.Pd[j]) * model.PTDF[i, j] for j in bus_idx)
        def Pg_max_inequality(model, i):
            return model.Pg[i] <= model.Pg_max[i]
        def Pg_min_inequality(model, i):
            return model.Pg[i] >= model.Pg_min[i]
        def Pf_decompose(model, i):
            return model.Pf[i] == model.Pf_pos[i] - model.Pf_neg[i] 
        def Pf_pos_upper(model, i):
            return model.Pf_pos[i] <= self.model.Dir[i] * self.model.M * model.Pf_max[i]
        def Pf_pos_lower(model, i):
            return 0 <= model.Pf_pos[i]
        def Pf_neg_upper(model, i):
            return model.Pf_neg[i] <= (1 - self.model.Dir[i]) * self.model.M * model.Pf_max[i]
        def Pf_neg_lower(model, i):
            return 0 <= model.Pf_neg[i] 
        def Carbon_equation(model, i):
            return model.CI[i] * (sum(model.Pg[j] * model.BusGen[i, j] for j in gen_idx) + sum(model.Pf_pos[k] for k in branch_idx if branch[k, 1] == i)) \
                == (sum(model.Pg[j] * model.BusGen[i, j] * model.E[j] for j in gen_idx) + sum(model.Pf_pos[k] * model.CI[branch[k, 0]] for k in branch_idx if branch[k, 1] == i)) 
        def Carbon_cap(model):
            return sum(model.CI[i] * model.Pd[i] for i in bus_idx) <= 100000000
        
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
    clmp = CLMP(net, solver='ipopt', verbose=True)
    clmp.solve()