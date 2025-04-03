import numpy as np
from pypower.api import case9, rundcpf, ppoption
from makeCEF import makeCEF
from pypower import idx_gen, idx_bus, idx_brch
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

"""
ppc 为 IEEE case；

E_G 为 N_G * 1 的发电机碳排放强度列向量；

var_id 中包括 [var_gen_id, var_load_id]，
    var_gen_id 和 var_load_id 又各自为包含随机功率注入节点的列表。
    这里还需要注意，平衡节点属于被迫随机的功率注入，实际上和真正的 RE、RL 之间存在共线性的关系。（DCPF 计算是线性的，所以才是共线性，
    如果采用 ACPF，就应该不存在共线性）

sample_size 为采样数量
"""


def find_array(arr_list, target):
    """
    用于查找当前流向特征在特征池 `pattern_pool` 的索引，同时也是特征编号
    :param arr_list: 特征池（嵌套列表存储）
    :param target: 当前特征（以潮流流向表示，2 * 2 * N_B 的 array）
    :return: 对于已存特征，返回索引；对于全新特征返回 -1
    """
    for idx, arr in enumerate(arr_list):
        if np.array_equal(arr[0], target):
            return idx
    return -1


class PCEF:
    def __init__(self, ppc, E_G, var_id, sample_size, expand=3):
        self.ppc = ppc
        self.E_G = E_G

        self.var_gen_list = var_id[0]
        self.var_load_list = var_id[1]
        self.var_num = len(self.var_gen_list) + len(self.var_load_list)

        self.sample_size = sample_size

        self.ppopt = ppoption(VERBOSE=0, OUT_ALL=0)

        self.pattern_numThrld = (self.var_num + 1) * 3

        self.load_row = [
            np.where(self.ppc['bus'][:, idx_bus.BUS_I] == load_id)[0][0]
            for load_id in self.var_load_list
        ]
        self.gen_row = [
            np.where(self.ppc['gen'][:, idx_gen.GEN_BUS] == gen_id)[0][0]
            for gen_id in self.var_gen_list
        ]

        # 识别平衡节点在 ‘bus’ 中所在行
        self.slack_bus_row = np.where(self.ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0][0]
        # 除 case300 外，内外索引都一致，因此 slack_bus 节点编号 = self.slack_bus_row + 1
        self.slack_bus = self.slack_bus_row + 1
        # 识别 slack bus 的等值发电机在 'gen' 中的行数
        self.slack_bus_gen = np.where(self.ppc['gen'][:, idx_gen.GEN_BUS] == self.slack_bus)[0][0]

        self.N = len(self.ppc['bus'])
        self.N_B = len(self.ppc['branch'])

        # 原始样本池
        self.raw_pool = []
        # 采样池（只包括典型样本，及用于回归的样本）
        self.sample_pool = []
        # D_m 系数存储
        # array:D_m   int:pattern_id
        self.D_m_pool = []
        # 结果池，从原始样本池中提取计算 PCEF
        self.res_pool = []
        # 特征池，保存了潮流流向特征，以及属于该特征的样本数
        # array:direction (2*2*N_B)     pattern_num
        self.pattern_pool = []
        # 原始结果池，即直接计算 CEF
        self.raw_res = []
        # 采样数放大倍数。
        self.expand = expand

    def generate_sample(self):
        """
        根据 RE RL 节点进行采样。还可以增加一个参数用于决定随机采样服从什么样的分布，
        先写为上下限确定的平均分布。
        :return: 直接更新 `self.raw_pool`
        """
        # RL 所在节点
        # array: var_load_default
        var_load_default = self.ppc['bus'][self.load_row, idx_bus.PD]
        upper = var_load_default * 1.5
        lower = var_load_default * 0.5
        load_data = [
            np.random.uniform(upper[i], lower[i], int(self.sample_size * self.expand))
            for i in range(len(self.var_load_list))
        ]
        load_data = np.array(load_data).reshape(-1, len(self.var_load_list))

        # RE 所在节点
        # array: var_gen_default
        var_gen_default = self.ppc['gen'][self.gen_row, idx_gen.PG]
        upper = var_gen_default * 1.5
        lower = var_gen_default * 0.5
        gen_data = [
            np.random.uniform(upper[i], lower[i], int(self.sample_size * self.expand))
            for i in range(len(self.var_gen_list))
        ]
        gen_data = np.array(gen_data).reshape(-1, len(self.var_gen_list))
        # gen_data  load_data
        s_data = np.hstack((gen_data, load_data))

        # 当供大于求，当前系统需要通过平衡节点倒送功率。平衡节点性质发生改变（对于当前系统相当于从功率注入变为功率输出），
        # 而平衡节点通常接有调频电厂，这就意味着该节点的碳势为设定值。节点性质发生改变，目前还不考虑此情况，因此在采样时将此种样本排除。
        count = 0
        for i in tqdm(range(int(self.sample_size * self.expand))):
            ppc_temp = deepcopy(self.ppc)
            # 将基准算例中的数值替换成随机结果
            ppc_temp['gen'][self.gen_row, idx_gen.PG] = s_data[i, :(len(self.var_gen_list))]
            ppc_temp['bus'][self.load_row, idx_bus.PD] = s_data[i, (len(self.var_gen_list)):]

            dcpf, _ = rundcpf(ppc_temp, self.ppopt)

            # 如果出现平衡节点功率倒送，就不选取该样本

            if dcpf['gen'][self.slack_bus_gen, idx_gen.PG] <= 0:
                continue
            else:
                self.raw_pool.append(dcpf)
                count += 1
            if count == self.sample_size:
                print('Complete Sampling')
                break

    def direction_check(self, dcpf):
        """
        导入 IEEE 组织形式的算例，根据潮流分布矩阵（包括正向和反向）中元素的正负号，作为潮流流向特征的表征方式
        :param dcpf: IEEE case
        :return: 2 * 2 * N_B 特征矩阵
        """
        brch_pf = dcpf['branch'][:, [idx_brch.F_BUS, idx_brch.T_BUS, idx_brch.PF, idx_brch.PT]]
        P_B = np.zeros((self.N, self.N))
        for l in range(self.N_B):
            fbus = int(brch_pf[l, 0] - 1)
            tbus = int(brch_pf[l, 1] - 1)
            P_B[fbus, tbus] = brch_pf[l, 2]
            P_B[tbus, fbus] = brch_pf[l, 3]
        # 根据元素正负确定流向，用`P_B`中正、负元素的位置确定特征
        forward_pos = np.array(np.where(P_B > 0))
        backward_pos = np.array(np.where(P_B < 0))
        direction = np.stack((forward_pos, backward_pos), axis=0)
        return direction

    def update_sample_pool(self, dcpf, direction, pattern_id):
        """
        更新 sample_pool 中的内容。sample_pool 中保存了所有将由于回归分析的样本
        :param dcpf:
        :param direction:
        :param pattern_id:
        :return:
        """
        E, snode, bus_trans = makeCEF(dcpf, self.E_G)
        sample = deepcopy(dcpf)
        sample['E'] = E
        sample['snode'] = snode
        sample['bus_trans'] = bus_trans
        sample['direction'] = direction
        sample['pattern_flag'] = pattern_id
        self.sample_pool.append(sample)

    def regression(self, pattern_id):
        """
        根据指定的随机功率注入节点，组织随机功率注入矩阵作为回归的决策变量，回归拟合各节点碳势
        :param pattern_id: 流向特征编号
        :return: 返回当前特征的 D_m
        """
        P_Sinj = np.zeros((self.pattern_numThrld, self.var_num + 1))
        samples = list(filter(lambda case: case['pattern_flag'] == pattern_id, self.sample_pool))
        # 组织随机变量矩阵
        # 前 len(self.var_gen_list) 个为包括平衡节点在内的随机发电注入
        # 后 len(self.var_load_list) 个为所有随机负荷（负）注入
        for i, s in enumerate(samples):
            # 随机发电注入包括平衡节点，因此是 RE 个数 +1
            P_Sinj[i, 0] = s['gen'][self.slack_bus_gen, idx_gen.PG]
            for g in range(len(self.var_gen_list)):
                P_Sinj[i, g+1] = s['gen'][self.gen_row[g], idx_gen.PG]
            for l in range(len(self.var_load_list)):
                P_Sinj[i, l+len(self.var_gen_list)+1] = -s['bus'][self.load_row[l], idx_bus.PD]

        E_tar = np.array([item['E'] for item in samples]).squeeze()

        model = LinearRegression()
        D_m = []
        for i in range(self.N):
            model.fit(P_Sinj, E_tar[:, i])
            temp_list = model.coef_
            temp_list = np.insert(temp_list, self.var_num+1, model.intercept_)
            D_m.append(temp_list)
        return np.array(D_m)

    def calculate(self):
        """
        主程序
        :return:
        """
        for i in tqdm(range(self.sample_size)):
            sample = self.raw_pool[i]
            # 生成流向特征，两个二维的矩阵
            direction = self.direction_check(sample)

            if len(self.sample_pool) == 0:
                self.update_sample_pool(sample, direction, 0)
                self.pattern_pool.append([direction, 1])
                # 作为第一个结果，更新到结果池中
                self.res_pool.append(self.sample_pool[-1])
            else:
                # 确定属于何种潮流流向特征
                pattern_id = find_array(self.pattern_pool, direction)
                if pattern_id != -1:
                    # 如果为已记录特征，记录数 +1
                    self.pattern_pool[pattern_id][1] += 1
                    # 如果数量刚好可以进行回归
                    if self.pattern_pool[pattern_id][1] == self.pattern_numThrld:
                        # 特征在特征池中的索引就是其代号 pattern_id
                        self.update_sample_pool(sample, direction, pattern_id)
                        self.res_pool.append(self.sample_pool[-1])
                        # 执行回归
                        D_m = self.regression(pattern_id)
                        self.D_m_pool.append({'D_m': D_m, 'pattern_id': pattern_id})
                        continue
                    # 如果数量不够，继续扩充样本池
                    elif self.pattern_pool[pattern_id][1] < self.pattern_numThrld:
                        self.update_sample_pool(sample, direction, pattern_id)
                        self.res_pool.append(self.sample_pool[-1])
                    # 如果数量足够，直接利用回归结果计算
                    elif self.pattern_pool[pattern_id][1] > self.pattern_numThrld:
                        # 随机发电注入包括平衡节点，因此是 RE 个数 +1
                        P_Sinj = np.zeros((self.var_num + 1,))

                        P_Sinj[0] = sample['gen'][0, idx_gen.PG]
                        for g in range(len(self.var_gen_list)):
                            P_Sinj[g + 1] = sample['gen'][self.gen_row[g], idx_gen.PG]
                        for l in range(len(self.var_load_list)):
                            P_Sinj[l + len(self.var_gen_list) + 1] = -sample['bus'][self.load_row[l], idx_bus.PD]

                        P_Sinj = np.insert(P_Sinj, self.var_num + 1, 1).reshape(-1, 1)

                        temp = list(filter(lambda D: D['pattern_id'] == pattern_id, self.D_m_pool))
                        D_m = temp[0]['D_m']
                        E = D_m @ P_Sinj

                        res = deepcopy(sample)
                        res['E'] = E
                        res['snode'] = None
                        res['bus_trans'] = None
                        res['direction'] = direction
                        res['pattern_flag'] = pattern_id
                        self.res_pool.append(res)
                # 如果是新特征
                else:
                    pattern_id = len(self.pattern_pool)
                    self.update_sample_pool(sample, direction, pattern_id)
                    self.pattern_pool.append([direction, 1])
                    self.res_pool.append(self.sample_pool[-1])

    def update_raw_res(self):
        for i in tqdm(range(self.sample_size)):
            sample = self.raw_pool[i]
            # direction = case.direction_check(sample)
            E, snode, bus_trans = makeCEF(sample, self.E_G)
            s = deepcopy(sample)
            s['E'] = E
            s['snode'] = snode
            s['bus_trans'] = bus_trans
            # s['direction'] = direction
            self.raw_res.append(s)

    def organize_CEI(self):
        # sample_size * N
        node_CEI_raw = []
        for i in tqdm(range(self.sample_size)):
            temp_list = []
            for j in range(self.N):
                temp_list.append(self.raw_res[i]['E'][j, 0])
            node_CEI_raw.append(temp_list)
        node_CEI_raw = np.array(node_CEI_raw)

        # sample_size * N
        node_CEI_res = []
        for i in tqdm(range(self.sample_size)):
            temp_list = []
            for j in range(self.N):
                temp_list.append(self.res_pool[i]['E'][j, 0])
            node_CEI_res.append(temp_list)
        node_CEI_res = np.array(node_CEI_res)

        return node_CEI_raw, node_CEI_res
