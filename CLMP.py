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
import os
from pathlib import Path

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
        self.gen_idx = range(gen.shape[0])
        bus = ppc['bus']
        self.bus_idx = range(bus.shape[0])
        branch = ppc['branch']
        self.branch_idx = range(branch.shape[0])
        ref_bus = np.where(bus[:, BUS_TYPE] == REF)[0][0]
        
        PTDF_matrix = makePTDF(baseMVA, bus, branch, slack=ref_bus,
                result_side=0, using_sparse_solver=True, branch_id=None, reduced=False)
        BusGen_matrix = np.zeros((len(self.bus_idx), len(self.gen_idx)))
        for i in self.bus_idx:
            for j in self.gen_idx:
                if gen[j, 0] == i:
                    BusGen_matrix[i, j] = 1
        Pf_max_matrix = np.zeros((len(self.bus_idx), len(self.bus_idx)))
        for i in self.branch_idx:
            Pf_max_matrix[int(branch[i, F_BUS]), int(branch[i, T_BUS])] = branch[i, 5]  #'RATE_A'
            Pf_max_matrix[int(branch[i, T_BUS]), int(branch[i, F_BUS])] = branch[i, 5]  #'RATE_A'
            
        def PTDF(model, i, j):
            return PTDF_matrix[i, j]

        def BusGen(model, i, j):
            return BusGen_matrix[i, j]

        def Pf_max(model, i, j):
            return Pf_max_matrix[i, j]
        
        Cost = self.net.poly_cost['cp1_eur_per_mw']    
        np.random.seed(100)
        Emission = pd.DataFrame(np.random.uniform(5, 10, len(self.gen_idx)), index=self.gen_idx, columns=['emission_factor'])

        # 创建模型参数            
        self.model.Pg_max = Param(self.gen_idx, initialize=gen[:, 8]) #'PMAX'
        self.model.Pg_min = Param(self.gen_idx, initialize=gen[:, 9]) #'PMIN'
        self.model.Pf_max = Param(self.bus_idx, self.bus_idx, initialize=Pf_max)
        self.model.Pd = Param(self.bus_idx, initialize=bus[:, 2]) #PD
        self.model.C = Param(self.gen_idx, initialize=Cost)
        self.model.E = Param(self.gen_idx, initialize=Emission) 
        self.model.BusGen = Param(self.bus_idx, self.gen_idx, initialize=BusGen)
        self.model.PTDF = Param(self.branch_idx, self.bus_idx, initialize=PTDF) 
        
        # 创建模型变量
        Pg = gen[:, 1] 
        p_flow = pd.DataFrame(np.concatenate((self.net.res_line.p_from_mw, self.net.res_trafo.p_hv_mw)), index=self.branch_idx, columns=['P'])
        p_flow['from_bus'] = branch[:, F_BUS]
        p_flow['to_bus'] = branch[:, T_BUS]
        Pf = np.zeros((len(self.bus_idx), len(self.bus_idx)))
        for index, row in p_flow.iterrows():
            Pf[int(row.from_bus), int(row.to_bus)] = row.P
            Pf[int(row.to_bus), int(row.from_bus)] = -row.P
                
        Sign = np.sign(Pf)
        Sign[Sign<0] = 0
        def Pf_pos(model, i, j):
            return np.clip(Pf, 0, None)[i, j]
        def Pf_neg(model, i, j):
            return - np.clip(Pf, None, 0)[i, j]
        def Dir(model, i, j):
            return Sign[i, j]
        
        self.model.Pg = Var(self.gen_idx, domain=Reals, initialize=Pg) #'PG'  
        self.model.Pf_pos = Var(self.bus_idx, self.bus_idx, domain=Reals, initialize=Pf_pos) 
        self.model.Pf_neg = Var(self.bus_idx, self.bus_idx, domain=Reals, initialize=Pf_neg)
        self.model.CI = Var(self.bus_idx, domain=Reals, initialize=[10]*len(self.bus_idx))
        self.model.Dir = Var(self.bus_idx, self.bus_idx, domain=Binary, initialize=Dir)
     
        # 创建模型约束
        # def power_balance(model, i):
        #     return sum(model.Pg[g] * model.BusGen[i, g] for g in self.gen_idx) + sum(model.Pf_pos[j, i] for j in self.bus_idx) == \
        #     sum(model.Pf_neg[j, i] for j in self.bus_idx) + model.Pd[i]
        def power_balance(model):
            return  sum(model.Pg[g] for g in self.gen_idx) == sum(model.Pd[i] for i in self.bus_idx)
        def Pf_equality(model, i, j):
            return model.Pf_pos[i, j] - model.Pf_neg[i, j] == sum(
                (sum(PTDF_matrix[b, k] for b in self.branch_idx if branch[b, F_BUS] == i and branch[b, T_BUS] == j) - \
                sum(PTDF_matrix[b, k] for b in self.branch_idx if branch[b, F_BUS] == j and branch[b, T_BUS] == i)) * \
                (sum(model.Pg[g] * model.BusGen[k, g] for g in self.gen_idx) - model.Pd[k]) for k in self.bus_idx)
        def Pg_max_inequality(model, g):
            return model.Pg[g] <= model.Pg_max[g]
        def Pg_min_inequality(model, g):
            return model.Pg_min[g] <= model.Pg[g] 
        def Pf_pos_upper(model, i, j):
            return model.Pf_pos[i, j] <= self.model.Dir[i, j] * model.Pf_max[i, j]
        def Pf_pos_lower(model, i, j):
            return 0 <= model.Pf_pos[i, j]
        def Pf_neg_upper(model, i, j):
            return model.Pf_neg[i, j] <= (1 - self.model.Dir[i, j]) * model.Pf_max[i, j]
        def Pf_neg_lower(model, i, j):
            return 0 <= model.Pf_neg[i, j] 
        def Carbon_equation(model, i):
            return model.CI[i] * (sum(model.Pg[g] * model.BusGen[i, g] for g in self.gen_idx) + \
                sum(model.Pf_pos[j, i] for j in self.bus_idx)) == \
                sum(model.Pg[g] * model.BusGen[i, g] * model.E[g] for g in self.gen_idx) + \
                sum(model.Pf_pos[j, i] * model.CI[j] for j in self.bus_idx)
        def Carbon_cap(model):
            return sum(model.CI[i] * model.Pd[i] for i in self.bus_idx) <= 1200
        
        self.model.power_balance = Constraint(rule=power_balance)
        self.model.Pf_equality = Constraint(self.bus_idx, self.bus_idx, rule=Pf_equality)
        self.model.Pg_max_inequality = Constraint(self.gen_idx, rule=Pg_max_inequality)
        self.model.Pg_min_inequality = Constraint(self.gen_idx, rule=Pg_min_inequality)
        self.model.Pf_pos_upper = Constraint(self.bus_idx, self.bus_idx, rule=Pf_pos_upper)
        self.model.Pf_pos_lower = Constraint(self.bus_idx, self.bus_idx, rule=Pf_pos_lower)
        self.model.Pf_neg_upper = Constraint(self.bus_idx, self.bus_idx, rule=Pf_neg_upper)
        self.model.Pf_neg_lower = Constraint(self.bus_idx, self.bus_idx, rule=Pf_neg_lower)
        self.model.Carbon_equation = Constraint(self.bus_idx, rule=Carbon_equation)
        self.model.Carbon_cap = Constraint(rule=Carbon_cap)
            
        # 创建模型目标
        self.model.objective = Objective(
            expr=sum(self.model.Pg[j] * self.model.C[j] for j in self.gen_idx), sense=minimize)

    def solve(self):
        self.model.dual = Suffix(direction=Suffix.IMPORT)
        opt = SolverFactory(self.solver)
        results = opt.solve(self.model, tee=self.verbose)
        print(results)
        return results
    
    def read_prime_solution(self):
        
        Pg = np.zeros(len(self.gen_idx))
        Pf_pos = np.zeros((len(self.bus_idx), len(self.bus_idx)))
        Pf_neg = np.zeros((len(self.bus_idx), len(self.bus_idx)))
        CI = np.zeros(len(self.bus_idx))
        Dir = np.zeros((len(self.bus_idx), len(self.bus_idx)))
        for g in self.gen_idx:
            Pg[g] = self.model.Pg[g].value
        for i in self.bus_idx:
            for j in self.bus_idx:
                Pf_pos[i, j] = self.model.Pf_pos[i, j].value
                Pf_neg[i, j] = self.model.Pf_neg[i, j].value
                Dir[i, j] = self.model.Dir[i, j].value
        for i in self.bus_idx:
            CI[i] = self.model.CI[i].value
        return Pg, Pf_pos, Pf_neg, CI, Dir
    
    def read_dual_solution(self):
        lambda_pb = self.model.dual[self.model.power_balance]
        lambda_pf = np.zeros((len(self.bus_idx), len(self.bus_idx)))
        for i in self.bus_idx:
            for j in self.bus_idx:
                lambda_pf[i, j] = self.model.dual[self.model.Pf_equality[i, j]]
        lambda_c = np.zeros(len(self.bus_idx))
        for i in self.bus_idx:
            lambda_c[i] = self.model.dual[self.model.Carbon_equation[i]]
        mu_g_upper = np.zeros(len(self.gen_idx))
        mu_g_lower = np.zeros(len(self.gen_idx))
        for g in self.gen_idx:
            mu_g_upper[g] = self.model.dual[self.model.Pg_max_inequality[g]]
            mu_g_lower[g] = self.model.dual[self.model.Pg_min_inequality[g]]
        mu_pos_upper = np.zeros((len(self.bus_idx), len(self.bus_idx)))
        mu_pos_lower = np.zeros((len(self.bus_idx), len(self.bus_idx)))
        for i in self.bus_idx:  
            for j in self.bus_idx:
                mu_pos_upper[i, j] = self.model.dual[self.model.Pf_pos_upper[i, j]]
                mu_pos_lower[i, j] = self.model.dual[self.model.Pf_pos_lower[i, j]]
        mu_neg_upper = np.zeros((len(self.bus_idx), len(self.bus_idx)))
        mu_neg_lower = np.zeros((len(self.bus_idx), len(self.bus_idx)))
        for i in self.bus_idx:  
            for j in self.bus_idx:
                mu_neg_upper[i, j] = self.model.dual[self.model.Pf_neg_upper[i, j]]
                mu_neg_lower[i, j] = self.model.dual[self.model.Pf_neg_lower[i, j]]
        mu_c = self.model.dual[self.model.Carbon_cap]
        return lambda_pb, lambda_pf, lambda_c, mu_g_upper, mu_g_lower, mu_pos_upper, mu_pos_lower, mu_neg_upper, mu_neg_lower, mu_c
    
if __name__ == "__main__":
    
    main_path = os.path.dirname(__file__)
    folder_path = os.path.join(main_path, 'data')

    if Path(folder_path).exists():
        pass
    else:
        os.mkdir(folder_path)
    # 创建一个示例网络
    net = pn.case30()
    # 创建一个CLMP模型
    clmp = CLMP(net, solver='ipopt', verbose=True)
    clmp.solve()
    
    Pg, Pf_pos, Pf_neg, CI, Dir = clmp.read_prime_solution()
    lambda_pb, lambda_pf, lambda_c, mu_g_upper, mu_g_lower, mu_pos_upper, mu_pos_lower, mu_neg_upper, mu_neg_lower, mu_c = clmp.read_dual_solution()
    # np.savetxt(os.path.join(folder_path,'intermediate','lamb da_pb.csv'), lambda_pb)
    np.savetxt(os.path.join(folder_path,'intermediate','lambda_pf.csv'), lambda_pf, delimiter=',')
    np.savetxt(os.path.join(folder_path,'intermediate','lambda_c.csv'), lambda_c, delimiter=',')
    np.savetxt(os.path.join(folder_path,'intermediate','mu_g_upper.csv'), mu_g_upper, delimiter=',')
    np.savetxt(os.path.join(folder_path,'intermediate','mu_g_lower.csv'), mu_g_lower, delimiter=',')
    np.savetxt(os.path.join(folder_path,'intermediate','mu_pos_upper.csv'), mu_pos_upper, delimiter=',')
    np.savetxt(os.path.join(folder_path,'intermediate','mu_pos_lower.csv'), mu_pos_lower, delimiter=',')
    np.savetxt(os.path.join(folder_path,'intermediate','mu_neg_upper.csv'), mu_neg_upper, delimiter=',')
    np.savetxt(os.path.join(folder_path,'intermediate','mu_neg_lower.csv'), mu_neg_lower, delimiter=',')
    print(lambda_pb)
    print(mu_c)
          
    # testing KKD
    dPf_pos = np.zeros((len(clmp.bus_idx), len(clmp.bus_idx)))
    dPf_neg = np.zeros((len(clmp.bus_idx), len(clmp.bus_idx)))
    for i in clmp.bus_idx:
        for j in clmp.bus_idx:
            dPf_pos[i, j] = lambda_pf[i, j] + mu_pos_upper[i, j] * Dir[i, j] - mu_pos_lower[i, j] * Dir[i, j] + lambda_c[j] * (CI[j] - CI[i])
            dPf_neg[i, j] = lambda_pf[i, j] + mu_neg_upper[i, j] * Dir[i, j] - mu_neg_lower[i, j] * Dir[i, j] + lambda_c[j] * (CI[j] - CI[i])
            
    print(np.max(dPf_pos))