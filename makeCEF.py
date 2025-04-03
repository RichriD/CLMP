from copy import deepcopy

import numpy as np
from pypower import idx_gen, idx_bus, idx_brch
from pypower.api import case9, rundcpf, ppoption, case300

ppc = case300()
ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
res, _ = rundcpf(ppc, ppopt)

E_G_default = np.random.rand(len(res['gen']), 1)
# E_G_default = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1]).reshape(6, 1)

"""矩阵形式的 dcpf CEF 计算方法，
同时默认的 dcpf 结果为 case9，E_G 为给定值
考虑了存在孤岛的情况，以及在case300中节点编号和索引不一致的情况
"""


def makeCEF(dcpf=res, E_G=E_G_default):
    # 节点数
    N = len(dcpf['bus'])
    # 支路数
    N_B = len(dcpf['branch'])
    # 发电机数
    N_G = len(dcpf['gen'])

    # 发电机出力
    # bus   Pg
    # gen_pf = deepcopy(dcpf['gen'][:, :2])
    # 支路潮流
    # fbus  tbus fp tp
    brch_pf = deepcopy(dcpf['branch'][:, [idx_brch.F_BUS, idx_brch.T_BUS, idx_brch.PF, idx_brch.PT]])

    # 检查节点编号和字段`bus`中节点数据所在索引是否一致，
    # 如果不一致，就需要设定内外索引，
    # 内索引表示原本设置的节点编号
    # 外索引为计算编号
    bus_trans = None
    if int(dcpf['bus'][-1, idx_bus.BUS_I]) != N:
        bus_in = dcpf['bus'][:, idx_bus.BUS_I].astype(int) # 算例bus序号
        bus_ex = np.array(range(bus_in.size)) # 计算序号
        bus_trans = np.column_stack((bus_in, bus_ex))

        for i in range(N_B):
            brch_pf[i, 0] = np.where(bus_trans[:, 0] == brch_pf[i, 0])[0][0]
            brch_pf[i, 1] = np.where(bus_trans[:, 0] == brch_pf[i, 1])[0][0]

    # 组织支路潮流为矩阵形式（array）
    # P_B (fbus, tbus)
    P_B = np.zeros((N, N))
    for l in range(N_B):
        fbus = int(brch_pf[l, 0] - 1)
        tbus = int(brch_pf[l, 1] - 1)

        # # P_B 中的元素应当全部为正实数
        P_B[fbus, tbus] = brch_pf[l, 2]
        P_B[tbus, fbus] = brch_pf[l, 3]
    P_B[P_B < 0] = 0

    # 节点负荷
    # 这里取负荷所在节点要求 bus 字段是按照顺序排列的
    load_bus = np.where(dcpf['bus'][:, idx_bus.PD] > 0)[0].tolist()
    N_L = len(load_bus)
    P_L = np.zeros((N, N_L))
    for d, val in enumerate(load_bus):
        P_L[val, d] = dcpf['bus'][val, idx_bus.PD]

    # 节点出力
    P_G = np.zeros((N, N_G))
    if int(dcpf['bus'][-1, idx_bus.BUS_I]) != N:
        for j in range(N_G):
            con_bus = np.where(bus_in == int(dcpf['gen'][j, idx_gen.GEN_BUS]))[0][0]
            P_G[con_bus, j] = dcpf['gen'][j, idx_gen.PG]
    else:
        for j in range(N_G):
            con_bus = int(dcpf['gen'][j, idx_gen.GEN_BUS]) - 1
            P_G[con_bus, j] = dcpf['gen'][j, idx_gen.PG]

    # 统计通量的时候，如果新能源不能完全消纳，只有通过负荷计算的才准确
    P_N = np.zeros((N, N))
    for i in range(N):
        P_N_ex = 0
        # 统计负荷
        load_list = np.where(P_L[i, :] > 0)[0].tolist()
        if len(load_list) > 0:
            P_N_ex += P_L[i, load_list].sum()

        # 统计支路流出
        brch_list = np.where(P_B[i, :] > 0)[0].tolist()  # i is fbus
        if len(brch_list) > 0:
            P_N_ex += P_B[i, brch_list].sum()
        P_N[i, i] = P_N_ex

    # 孤岛节点应该按照通量来确定
    zero_row = np.where(np.all(P_N == 0, axis=1))[0]
    if len(zero_row) == 0:
        E = np.linalg.inv(P_N - P_B.T) @ P_G @ E_G
        return E.reshape(N, 1), None, bus_trans
    elif len(zero_row) > 0:
        # print('Single Node Exist!')
        # print('Single Node:', zero_row + 1)

        # 去除孤岛节点
        non_zero_row_mask = ~np.all(P_N == 0, axis=1)
        non_zero_col_mask = ~np.all(P_N == 0, axis=0)
        P_B = P_B[non_zero_row_mask]
        P_B = P_B[:, non_zero_col_mask]
        P_N = P_N[non_zero_row_mask]
        P_N = P_N[:, non_zero_col_mask]
        P_G = P_G[non_zero_row_mask]

        E = np.linalg.inv(P_N - P_B.T) @ P_G @ E_G
        E = np.insert(E, zero_row, 0)
        return E.reshape(N, 1), zero_row, bus_trans


# if __name__ == '__main__':
#     makeCEF()
