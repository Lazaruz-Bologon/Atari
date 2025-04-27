import numpy as np
import torch
import torch.nn as nn
class WilsonCowanConnectivity:
    """
    Wilson-Cowan 神经群落网络的专属连接结构
    基于神经科学原理设计的连接模式
    """
    def __init__(self, hidden_size=64, output_size=4, 
                 excitatory_ratio=0.8, connectivity='small-world', 
                 sparsity=0.2, seed=42):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.excitatory_ratio = excitatory_ratio
        self.connectivity_type = connectivity
        self.sparsity = sparsity
        self._rng = np.random.RandomState(seed)
        # 计算兴奋性和抑制性神经元数量
        self.num_excitatory = int(hidden_size * excitatory_ratio)
        self.num_inhibitory = hidden_size - self.num_excitatory
        # 索引列表
        self._excitatory_neurons = list(range(0, self.num_excitatory))
        self._inhibitory_neurons = list(range(self.num_excitatory, hidden_size))
        # 连接矩阵初始化
        self.w_ee = None  # 兴奋到兴奋
        self.w_ei = None  # 兴奋到抑制
        self.w_ie = None  # 抑制到兴奋
        self.w_ii = None  # 抑制到抑制
        self.input_weights_e = None  # 输入到兴奋性
        self.input_weights_i = None  # 输入到抑制性
        # 神经科学特性参数
        self.inhibitory_density_factor = 1.5  # 抑制性神经元通常有更广泛的连接
        self.excitatory_strength = (0.1, 0.3)  # 兴奋性连接的强度范围
        self.inhibitory_strength = (-0.4, -0.1)  # 抑制性连接的强度范围
        # 距离依赖参数
        self.distance_lambda = 2.0  # 控制连接概率随距离衰减的参数
        # E-I平衡参数 (Dale法则: 神经元要么全兴奋，要么全抑制)
        self.balance_factor = 0.8  # 控制E-I平衡水平
    def build(self, input_size):
        """构建连接矩阵"""
        # 初始化连接矩阵为零矩阵
        self.w_ee = np.zeros((self.num_excitatory, self.num_excitatory))
        self.w_ei = np.zeros((self.num_excitatory, self.num_inhibitory))
        self.w_ie = np.zeros((self.num_inhibitory, self.num_excitatory))
        self.w_ii = np.zeros((self.num_inhibitory, self.num_inhibitory))
        # 根据指定的连接模式创建连接
        if self.connectivity_type == 'full':
            self._create_fully_connected()
        elif self.connectivity_type == 'small-world':
            self._create_small_world()
        elif self.connectivity_type == 'modular':
            self._create_modular()
        elif self.connectivity_type == 'distance-dependent':
            self._create_distance_dependent()
        elif self.connectivity_type == 'hierarchical':
            self._create_hierarchical()
        else:  # random
            self._create_random()
        # 实施Dale法则
        self._enforce_dale_law()
        # 确保E-I平衡
        self._ensure_ei_balance()
        # 为输入创建结构化连接模式
        self.input_weights_e = self._create_structured_input_weights(input_size, self.num_excitatory, True)
        self.input_weights_i = self._create_structured_input_weights(input_size, self.num_inhibitory, False)
        # 转换为PyTorch张量
        self.w_ee = torch.FloatTensor(self.w_ee)
        self.w_ei = torch.FloatTensor(self.w_ei)
        self.w_ie = torch.FloatTensor(self.w_ie)
        self.w_ii = torch.FloatTensor(self.w_ii)
        self.input_weights_e = torch.FloatTensor(self.input_weights_e)
        self.input_weights_i = torch.FloatTensor(self.input_weights_i)
        return {
            'w_ee': self.w_ee,
            'w_ei': self.w_ei,
            'w_ie': self.w_ie,
            'w_ii': self.w_ii,
            'input_e': self.input_weights_e,
            'input_i': self.input_weights_i
        }
    def _create_fully_connected(self):
        """创建全连接网络，但带有生物学特性"""
        # 兴奋性到兴奋性连接：局部增强，远程弱化
        for i in range(self.num_excitatory):
            for j in range(self.num_excitatory):
                if i != j:  # 避免自连接
                    distance = abs(i - j) / self.num_excitatory
                    strength = self._rng.uniform(self.excitatory_strength[0], 
                                                self.excitatory_strength[1])
                    if distance > 0.5:  # 远距离连接较弱
                        strength *= 0.5
                    self.w_ee[i, j] = strength
        # 兴奋性到抑制性连接：较强
        for i in range(self.num_excitatory):
            for j in range(self.num_inhibitory):
                strength = self._rng.uniform(self.excitatory_strength[0] * 1.2, 
                                            self.excitatory_strength[1] * 1.2)
                self.w_ei[i, j] = strength
        # 抑制性到兴奋性连接：反馈抑制
        for i in range(self.num_inhibitory):
            for j in range(self.num_excitatory):
                strength = self._rng.uniform(self.inhibitory_strength[0], 
                                            self.inhibitory_strength[1])
                self.w_ie[i, j] = strength
        # 抑制性到抑制性连接：抑制抑制
        for i in range(self.num_inhibitory):
            for j in range(self.num_inhibitory):
                if i != j:  # 避免自连接
                    # 抑制性神经元之间的抑制造成了脱抑制效应
                    strength = self._rng.uniform(self.inhibitory_strength[0] * 0.7, 
                                                self.inhibitory_strength[1] * 0.7)
                    self.w_ii[i, j] = strength
    def _create_small_world(self):
        """创建小世界网络连接模式"""
        # 实现Watts-Strogatz小世界网络模型，增加神经科学特性
        # 计算每个神经元的邻居数量
        k_e = max(int((1 - self.sparsity) * self.num_excitatory / 2) * 2, 2)  # 兴奋性
        k_i = max(int((1 - self.sparsity) * self.num_inhibitory / 2) * 2, 2)  # 抑制性
        # 1. 创建环状近邻连接
        # 兴奋性到兴奋性的环状连接
        for i in range(self.num_excitatory):
            for j in range(1, k_e // 2 + 1):
                # 连接前面和后面的k/2个邻居
                ahead = (i + j) % self.num_excitatory
                behind = (i - j) % self.num_excitatory
                self.w_ee[i, ahead] = self._rng.uniform(self.excitatory_strength[0], 
                                                      self.excitatory_strength[1])
                self.w_ee[i, behind] = self._rng.uniform(self.excitatory_strength[0], 
                                                       self.excitatory_strength[1])
        # 抑制性到抑制性的环状连接
        for i in range(self.num_inhibitory):
            for j in range(1, k_i // 2 + 1):
                # 连接前面和后面的k/2个邻居
                ahead = (i + j) % self.num_inhibitory
                behind = (i - j) % self.num_inhibitory
                self.w_ii[i, ahead] = self._rng.uniform(self.inhibitory_strength[0], 
                                                      self.inhibitory_strength[1])
                self.w_ii[i, behind] = self._rng.uniform(self.inhibitory_strength[0], 
                                                       self.inhibitory_strength[1])
        # 兴奋性到抑制性的连接 (更稀疏)
        for i in range(self.num_excitatory):
            targets = self._rng.choice(self.num_inhibitory, 
                                     size=max(1, int(self.num_inhibitory * (1 - self.sparsity * 1.2))), 
                                     replace=False)
            for j in targets:
                self.w_ei[i, j] = self._rng.uniform(self.excitatory_strength[0], 
                                                  self.excitatory_strength[1])
        # 抑制性到兴奋性的连接 (广泛)
        for i in range(self.num_inhibitory):
            targets = self._rng.choice(self.num_excitatory, 
                                     size=max(1, int(self.num_excitatory * (1 - self.sparsity * 0.8))), 
                                     replace=False)
            for j in targets:
                self.w_ie[i, j] = self._rng.uniform(self.inhibitory_strength[0], 
                                                  self.inhibitory_strength[1])
        # 2. 重连部分连接，形成"小世界"特性
        p_rewire = 0.15  # 重连概率
        # 重连兴奋性连接
        for i in range(self.num_excitatory):
            for j in range(self.num_excitatory):
                if self.w_ee[i, j] != 0 and self._rng.random() < p_rewire:
                    # 断开原有连接
                    old_weight = self.w_ee[i, j]
                    self.w_ee[i, j] = 0
                    # 随机选择新目标
                    new_target = self._rng.randint(0, self.num_excitatory)
                    attempts = 0
                    while (new_target == i or self.w_ee[i, new_target] != 0) and attempts < 10:
                        new_target = self._rng.randint(0, self.num_excitatory)
                        attempts += 1
                    # 建立新连接或恢复原连接
                    if attempts < 10:
                        self.w_ee[i, new_target] = old_weight
                    else:
                        self.w_ee[i, j] = old_weight
        # 重连抑制性连接
        for i in range(self.num_inhibitory):
            for j in range(self.num_inhibitory):
                if self.w_ii[i, j] != 0 and self._rng.random() < p_rewire:
                    # 断开原有连接
                    old_weight = self.w_ii[i, j]
                    self.w_ii[i, j] = 0
                    # 随机选择新目标
                    new_target = self._rng.randint(0, self.num_inhibitory)
                    attempts = 0
                    while (new_target == i or self.w_ii[i, new_target] != 0) and attempts < 10:
                        new_target = self._rng.randint(0, self.num_inhibitory)
                        attempts += 1
                    # 建立新连接或恢复原连接
                    if attempts < 10:
                        self.w_ii[i, new_target] = old_weight
                    else:
                        self.w_ii[i, j] = old_weight
    def _create_modular(self):
        """创建模块化网络连接模式"""
        # 确定模块数量
        num_modules = min(4, self.hidden_size // 16)
        e_per_module = self.num_excitatory // num_modules
        i_per_module = self.num_inhibitory // num_modules
        # 创建模块内连接 (密集)
        for m in range(num_modules):
            # 兴奋性模块
            e_start = m * e_per_module
            e_end = min((m + 1) * e_per_module, self.num_excitatory)
            # 模块内兴奋性连接
            for i in range(e_start, e_end):
                for j in range(e_start, e_end):
                    if i != j and self._rng.random() > self.sparsity * 0.5:
                        self.w_ee[i, j] = self._rng.uniform(self.excitatory_strength[0], 
                                                          self.excitatory_strength[1])
            # 抑制性模块
            i_start = m * i_per_module
            i_end = min((m + 1) * i_per_module, self.num_inhibitory)
            # 模块内抑制性连接
            for i in range(i_start, i_end):
                for j in range(i_start, i_end):
                    if i != j and self._rng.random() > self.sparsity * 0.5:
                        self.w_ii[i, j] = self._rng.uniform(self.inhibitory_strength[0], 
                                                          self.inhibitory_strength[1])
            # 模块内兴奋到抑制连接
            for i in range(e_start, e_end):
                for j in range(i_start, i_end):
                    if self._rng.random() > self.sparsity * 0.5:
                        self.w_ei[i, j] = self._rng.uniform(self.excitatory_strength[0], 
                                                          self.excitatory_strength[1])
            # 模块内抑制到兴奋连接
            for i in range(i_start, i_end):
                for j in range(e_start, e_end):
                    if self._rng.random() > self.sparsity * 0.5:
                        self.w_ie[i, j] = self._rng.uniform(self.inhibitory_strength[0], 
                                                          self.inhibitory_strength[1])
        # 创建模块间连接 (稀疏)
        for m1 in range(num_modules):
            for m2 in range(num_modules):
                if m1 == m2:  # 跳过模块内连接
                    continue
                e_start1 = m1 * e_per_module
                e_end1 = min((m1 + 1) * e_per_module, self.num_excitatory)
                i_start1 = m1 * i_per_module
                i_end1 = min((m1 + 1) * i_per_module, self.num_inhibitory)
                e_start2 = m2 * e_per_module
                e_end2 = min((m2 + 1) * e_per_module, self.num_excitatory)
                i_start2 = m2 * i_per_module
                i_end2 = min((m2 + 1) * i_per_module, self.num_inhibitory)
                # 模块间兴奋性连接 (稀疏)
                for i in range(e_start1, e_end1):
                    # 选择少量目标
                    num_targets = max(1, int((e_end2 - e_start2) * (1 - self.sparsity * 2)))
                    targets = self._rng.choice(range(e_start2, e_end2), size=num_targets, replace=False)
                    for j in targets:
                        # 模块间连接强度较弱
                        strength = self._rng.uniform(self.excitatory_strength[0], 
                                                  self.excitatory_strength[1]) * 0.7
                        self.w_ee[i, j] = strength
                # 模块间抑制性连接 (更广泛)
                for i_idx in range(i_start1, i_end1):
                    # 抑制性神经元有更广泛的模块间投射
                    num_targets = max(1, int((e_end2 - e_start2) * (1 - self.sparsity * 1.5)))
                    targets = self._rng.choice(range(e_start2, e_end2), size=num_targets, replace=False)
                    for j in targets:
                        strength = self._rng.uniform(self.inhibitory_strength[0], 
                                                self.inhibitory_strength[1]) * 0.8
                        # 使用正确的抑制性神经元索引
                        self.w_ie[i_idx, j] = strength
    def _create_distance_dependent(self):
        """创建基于距离的连接模式"""
        # 兴奋性到兴奋性连接
        for i in range(self.num_excitatory):
            for j in range(self.num_excitatory):
                if i == j:  # 避免自连接
                    continue
                # 计算归一化距离
                distance = abs(i - j) / self.num_excitatory
                # 使用指数衰减计算连接概率
                p_connect = np.exp(-distance * self.distance_lambda)
                # 连接概率随稀疏度调整
                p_connect *= (1 - self.sparsity)
                if self._rng.random() < p_connect:
                    # 连接强度随距离衰减
                    strength = self._rng.uniform(self.excitatory_strength[0], 
                                               self.excitatory_strength[1])
                    strength *= max(0.3, 1.0 - 0.7 * distance)
                    self.w_ee[i, j] = strength
        # 抑制性到抑制性连接
        for i in range(self.num_inhibitory):
            for j in range(self.num_inhibitory):
                if i == j:  # 避免自连接
                    continue
                # 计算归一化距离
                distance = abs(i - j) / self.num_inhibitory
                # 抑制性神经元有更广泛的连接范围
                p_connect = np.exp(-distance * (self.distance_lambda * 0.7))
                p_connect *= (1 - self.sparsity) * self.inhibitory_density_factor
                if self._rng.random() < p_connect:
                    strength = self._rng.uniform(self.inhibitory_strength[0], 
                                               self.inhibitory_strength[1])
                    strength *= max(0.3, 1.0 - 0.5 * distance)  # 抑制衰减较慢
                    self.w_ii[i, j] = strength
        # 兴奋性到抑制性连接
        for i in range(self.num_excitatory):
            for j in range(self.num_inhibitory):
                # 计算到抑制性群体的距离
                position_ratio = i / self.num_excitatory - j / self.num_inhibitory
                distance = min(abs(position_ratio), 1 - abs(position_ratio))
                p_connect = np.exp(-distance * self.distance_lambda)
                p_connect *= (1 - self.sparsity)
                if self._rng.random() < p_connect:
                    strength = self._rng.uniform(self.excitatory_strength[0], 
                                               self.excitatory_strength[1])
                    self.w_ei[i, j] = strength
        # 抑制性到兴奋性连接 (更广泛)
        for i in range(self.num_inhibitory):
            # 抑制性神经元通常有更广泛的投射
            num_targets = max(1, int(self.num_excitatory * (1 - self.sparsity * 0.7)))
            targets = self._rng.choice(self.num_excitatory, size=num_targets, replace=False)
            for j in targets:
                # 衰减较少
                strength = self._rng.uniform(self.inhibitory_strength[0], 
                                           self.inhibitory_strength[1])
                self.w_ie[i, j] = strength
    def _create_hierarchical(self):
        """创建层次化的连接模式"""
        # 确定层次数量
        num_layers = min(4, self.hidden_size // 16)
        # 计算每层的神经元数量
        e_per_layer = self.num_excitatory // num_layers
        i_per_layer = self.num_inhibitory // num_layers
        # 为每个神经元分配层
        e_layer_assignment = []
        for l in range(num_layers):
            e_layer_assignment.extend([l] * e_per_layer)
        # 处理余数
        if len(e_layer_assignment) < self.num_excitatory:
            e_layer_assignment.extend([num_layers-1] * (self.num_excitatory - len(e_layer_assignment)))
        i_layer_assignment = []
        for l in range(num_layers):
            i_layer_assignment.extend([l] * i_per_layer)
        # 处理余数
        if len(i_layer_assignment) < self.num_inhibitory:
            i_layer_assignment.extend([num_layers-1] * (self.num_inhibitory - len(i_layer_assignment)))
        # 层内兴奋性连接 (密集)
        for i in range(self.num_excitatory):
            i_layer = e_layer_assignment[i]
            for j in range(self.num_excitatory):
                if i == j:  # 避免自连接
                    continue
                j_layer = e_layer_assignment[j]
                if i_layer == j_layer:  # 同层连接密集
                    if self._rng.random() > self.sparsity * 0.5:
                        self.w_ee[i, j] = self._rng.uniform(self.excitatory_strength[0], 
                                                          self.excitatory_strength[1])
        # 层内抑制性连接
        for i in range(self.num_inhibitory):
            i_layer = i_layer_assignment[i]
            for j in range(self.num_inhibitory):
                if i == j:  # 避免自连接
                    continue
                j_layer = i_layer_assignment[j]
                if i_layer == j_layer:  # 同层连接
                    if self._rng.random() > self.sparsity * 0.5:
                        self.w_ii[i, j] = self._rng.uniform(self.inhibitory_strength[0], 
                                                          self.inhibitory_strength[1])
        # 层间兴奋性连接 (向上投射，前馈)
        for i in range(self.num_excitatory):
            i_layer = e_layer_assignment[i]
            for j in range(self.num_excitatory):
                if i == j:
                    continue
                j_layer = e_layer_assignment[j]
                if j_layer > i_layer:  # 向上投射
                    p_connect = 0.3 * (1 - self.sparsity) * (1.0 - 0.2 * (j_layer - i_layer))
                    if self._rng.random() < p_connect:
                        self.w_ee[i, j] = self._rng.uniform(self.excitatory_strength[0], 
                                                          self.excitatory_strength[1])
        # 层间抑制性连接 (双向，但向下更多)
        for i in range(self.num_inhibitory):
            i_layer = i_layer_assignment[i]
            for j in range(self.num_excitatory):
                j_layer = e_layer_assignment[j]
                if j_layer < i_layer:  # 向下抑制
                    p_connect = 0.4 * (1 - self.sparsity) * (1.0 - 0.1 * (i_layer - j_layer))
                    if self._rng.random() < p_connect:
                        self.w_ie[i, j] = self._rng.uniform(self.inhibitory_strength[0], 
                                                          self.inhibitory_strength[1])
                else:  # 向上抑制 (较弱)
                    p_connect = 0.2 * (1 - self.sparsity) * (1.0 - 0.3 * (j_layer - i_layer))
                    if self._rng.random() < p_connect:
                        self.w_ie[i, j] = self._rng.uniform(self.inhibitory_strength[0] * 0.7, 
                                                          self.inhibitory_strength[1] * 0.7)
        # 兴奋性到抑制性的层间连接
        for i in range(self.num_excitatory):
            i_layer = e_layer_assignment[i]
            # 每个兴奋性神经元连接到几个每层的抑制性神经元
            for l in range(num_layers):
                layer_i_start = sum(1 for x in i_layer_assignment if x < l)
                layer_i_end = sum(1 for x in i_layer_assignment if x <= l)
                if layer_i_end > layer_i_start:
                    # 选择该层的抑制性神经元进行连接
                    num_targets = max(1, int((layer_i_end - layer_i_start) * (1 - self.sparsity)))
                    targets = self._rng.choice(
                        range(layer_i_start, layer_i_end),
                        size=min(num_targets, layer_i_end - layer_i_start),
                        replace=False
                    )
                    for j in targets:
                        self.w_ei[i, j] = self._rng.uniform(self.excitatory_strength[0], 
                                                          self.excitatory_strength[1])
    def _create_random(self):
        """创建随机连接模式，但保持生物学合理性"""
        # 兴奋性到兴奋性随机连接
        for i in range(self.num_excitatory):
            # 确定连接数
            num_connections = int(self.num_excitatory * (1 - self.sparsity))
            targets = self._rng.choice(
                [j for j in range(self.num_excitatory) if j != i],  # 排除自连接
                size=min(num_connections, self.num_excitatory - 1),
                replace=False
            )
            for j in targets:
                self.w_ee[i, j] = self._rng.uniform(self.excitatory_strength[0], 
                                                  self.excitatory_strength[1])
        # 抑制性到抑制性随机连接
        for i in range(self.num_inhibitory):
            num_connections = int(self.num_inhibitory * (1 - self.sparsity) * self.inhibitory_density_factor)
            targets = self._rng.choice(
                [j for j in range(self.num_inhibitory) if j != i],
                size=min(num_connections, self.num_inhibitory - 1),
                replace=False
            )
            for j in targets:
                self.w_ii[i, j] = self._rng.uniform(self.inhibitory_strength[0], 
                                                  self.inhibitory_strength[1])
        # 兴奋性到抑制性随机连接
        for i in range(self.num_excitatory):
            num_connections = int(self.num_inhibitory * (1 - self.sparsity))
            targets = self._rng.choice(
                range(self.num_inhibitory),
                size=min(num_connections, self.num_inhibitory),
                replace=False
            )
            for j in targets:
                self.w_ei[i, j] = self._rng.uniform(self.excitatory_strength[0], 
                                                  self.excitatory_strength[1])
        # 抑制性到兴奋性随机连接
        for i in range(self.num_inhibitory):
            num_connections = int(self.num_excitatory * (1 - self.sparsity) * self.inhibitory_density_factor)
            targets = self._rng.choice(
                range(self.num_excitatory),
                size=min(num_connections, self.num_excitatory),
                replace=False
            )
            for j in targets:
                self.w_ie[i, j] = self._rng.uniform(self.inhibitory_strength[0], 
                                                  self.inhibitory_strength[1])
    def _enforce_dale_law(self):
        """实施Dale法则: 神经元要么全是兴奋性输出，要么全是抑制性输出"""
        # 确保兴奋性神经元只有正权重
        for i in range(self.num_excitatory):
            for j in range(self.num_excitatory):
                if self.w_ee[i, j] < 0:
                    self.w_ee[i, j] = -self.w_ee[i, j]
            for j in range(self.num_inhibitory):
                if self.w_ei[i, j] < 0:
                    self.w_ei[i, j] = -self.w_ei[i, j]
        # 确保抑制性神经元只有负权重
        for i in range(self.num_inhibitory):
            for j in range(self.num_excitatory):
                if self.w_ie[i, j] > 0:
                    self.w_ie[i, j] = -self.w_ie[i, j]
            for j in range(self.num_inhibitory):
                if self.w_ii[i, j] > 0:
                    self.w_ii[i, j] = -self.w_ii[i, j]
    def _ensure_ei_balance(self):
        """确保兴奋-抑制平衡"""
        # 计算每个兴奋性神经元接收的总输入
        for j in range(self.num_excitatory):
            # 计算兴奋性输入
            total_excitatory = sum(self.w_ee[i, j] for i in range(self.num_excitatory) if self.w_ee[i, j] > 0)
            # 计算抑制性输入
            total_inhibitory = sum(-self.w_ie[i, j] for i in range(self.num_inhibitory) if self.w_ie[i, j] < 0)
            # 计算E/I比例
            if total_inhibitory > 0:
                ei_ratio = total_excitatory / total_inhibitory
            else:
                ei_ratio = float('inf')
            # 如果比例失衡，调整权重
            if total_inhibitory > 0:  # 避免除以零
                target_ratio = self.balance_factor
                if ei_ratio > target_ratio * 1.5:  # 兴奋过强
                    # 增强抑制
                    scale = target_ratio / ei_ratio
                    for i in range(self.num_inhibitory):
                        if self.w_ie[i, j] < 0:
                            self.w_ie[i, j] *= 1.2
                elif ei_ratio < target_ratio * 0.5:  # 抑制过强
                    # 削弱抑制
                    for i in range(self.num_inhibitory):
                        if self.w_ie[i, j] < 0:
                            self.w_ie[i, j] *= 0.8
    def _create_structured_input_weights(self, input_size, output_size, is_excitatory):
        """创建结构化的输入权重"""
        weights = np.zeros((input_size, output_size))
        # 输入分组，每组优先连接到特定的输出神经元
        num_groups = min(3, input_size)
        outputs_per_group = output_size // num_groups
        for i in range(input_size):
            group = i % num_groups
            group_start = group * outputs_per_group
            group_end = min((group + 1) * outputs_per_group, output_size)
            # 主要组连接 (密集)
            for j in range(group_start, group_end):
                if self._rng.random() > self.sparsity * 0.5:  # 主要组稠密连接
                    if is_excitatory:
                        weights[i, j] = self._rng.uniform(0.1, 0.3)
                    else:
                        weights[i, j] = self._rng.uniform(-0.3, -0.1)
            # 次要组连接 (稀疏)
            other_outputs = list(range(0, group_start)) + list(range(group_end, output_size))
            if other_outputs:
                # 随机选择少量次要目标
                num_targets = max(1, int(len(other_outputs) * (1 - self.sparsity * 2)))
                targets = self._rng.choice(other_outputs, size=min(num_targets, len(other_outputs)), replace=False)
                for j in targets:
                    # 次要组连接强度较弱
                    if is_excitatory:
                        weights[i, j] = self._rng.uniform(0.05, 0.15)
                    else:
                        weights[i, j] = self._rng.uniform(-0.15, -0.05)
        return weights
class WilsonCowanCell(nn.Module):
    """Wilson-Cowan 神经群落模型的单元实现，使用专用连接结构"""
    def __init__(self, input_size, output_size, hidden_size=64, 
                 excitatory_ratio=0.8, connectivity='small-world', 
                 sparsity=0.2, dt=0.1, use_rk4=True, seed=42):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.excitatory_ratio = excitatory_ratio
        self.dt = dt
        self.use_rk4 = use_rk4
        self.seed = seed
        # 计算兴奋性和抑制性神经元数量
        self.num_excitatory = int(hidden_size * excitatory_ratio)
        self.num_inhibitory = hidden_size - self.num_excitatory
        # 创建连接结构
        connectivity_builder = WilsonCowanConnectivity(
            hidden_size=hidden_size,
            output_size=output_size,
            excitatory_ratio=excitatory_ratio,
            connectivity=connectivity,
            sparsity=sparsity,
            seed=seed
        )
        # 构建连接矩阵
        connection_matrices = connectivity_builder.build(input_size)
        # 注册连接权重为参数
        self.w_ee = nn.Parameter(connection_matrices['w_ee'])
        self.w_ei = nn.Parameter(connection_matrices['w_ei'])
        self.w_ie = nn.Parameter(connection_matrices['w_ie'])
        self.w_ii = nn.Parameter(connection_matrices['w_ii'])
        self.input_weights_e = nn.Parameter(connection_matrices['input_e'])
        self.input_weights_i = nn.Parameter(connection_matrices['input_i'])
        # 模型参数
        self.tau_e = nn.Parameter(torch.ones(self.num_excitatory) * 10.0)
        self.tau_i = nn.Parameter(torch.ones(self.num_inhibitory) * 20.0)
        self.a_e = nn.Parameter(torch.ones(self.num_excitatory) * 1.2)
        self.a_i = nn.Parameter(torch.ones(self.num_inhibitory) * 1.0)
        self.theta_e = nn.Parameter(torch.ones(self.num_excitatory) * 4.0)
        self.theta_i = nn.Parameter(torch.ones(self.num_inhibitory) * 3.7)
        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
    def _sigmoid(self, x, a, theta):
        return 1 / (1 + torch.exp(-a * (x - theta)))
    def _wc_dynamics(self, state, inputs):
        # 分离兴奋性和抑制性群体
        E = state[:, :self.num_excitatory]
        I = state[:, self.num_excitatory:]
        # 计算群体间互动
        ee_interaction = torch.matmul(E, self.w_ee)  # 兴奋到兴奋
        ie_interaction = torch.matmul(I, self.w_ie)  # 抑制到兴奋
        ei_interaction = torch.matmul(E, self.w_ei)  # 兴奋到抑制
        ii_interaction = torch.matmul(I, self.w_ii)  # 抑制到抑制
        # 外部输入投影
        P_ext = torch.matmul(inputs, self.input_weights_e)
        Q_ext = torch.matmul(inputs, self.input_weights_i)
        # 总输入
        P = ee_interaction + ie_interaction + P_ext  # 到兴奋群体的总输入
        Q = ei_interaction + ii_interaction + Q_ext  # 到抑制群体的总输入
        # 计算活性函数
        S_e = self._sigmoid(P, self.a_e, self.theta_e)
        S_i = self._sigmoid(Q, self.a_i, self.theta_i)
        # 计算微分方程
        dE = (-E + S_e) / self.tau_e
        dI = (-I + S_i) / self.tau_i
        return torch.cat([dE, dI], dim=1)
    def _euler_step(self, state, inputs):
        """欧拉法进行数值积分"""
        dstate = self._wc_dynamics(state, inputs)
        next_state = state + self.dt * dstate
        return next_state
    def _rk4_step(self, state, inputs):
        """四阶龙格库塔法进行更精确的数值积分"""
        k1 = self._wc_dynamics(state, inputs)
        k2 = self._wc_dynamics(state + 0.5 * self.dt * k1, inputs)
        k3 = self._wc_dynamics(state + 0.5 * self.dt * k2, inputs)
        k4 = self._wc_dynamics(state + self.dt * k3, inputs)
        next_state = state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return next_state
    def forward(self, x, state=None):
        """前向传播，处理输入序列"""
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device
        # 初始化状态
        if state is None:
            state = torch.cat([
                torch.ones(batch_size, self.num_excitatory, device=device) * 0.1,
                torch.ones(batch_size, self.num_inhibitory, device=device) * 0.1
            ], dim=1)
        # 处理序列
        outputs = []
        for t in range(seq_len):
            current_input = x[:, t]
            # 使用选择的积分方法更新状态
            if self.use_rk4:
                state = self._rk4_step(state, current_input)
            else:
                state = self._euler_step(state, current_input)
            # 计算输出
            output = self.output_layer(state)
            outputs.append(output)
        # 堆叠时间步输出
        outputs = torch.stack(outputs, dim=1)
        return outputs, state