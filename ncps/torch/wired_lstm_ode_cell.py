import numpy as np
import torch
from torch import nn

class SparseLinearLSTMODECell(nn.Module):
    def __init__(self, input_size, hidden_size, input_sparsity=None, 
                 ode_unfolds=6, epsilon=1e-8, ode_method='mixed'):
        super(SparseLinearLSTMODECell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._ode_method = ode_method  # 'euler', 'midpoint', 'rk4', 'analytic', 'mixed'
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.time_constants = nn.Parameter(torch.Tensor(hidden_size))
        self.has_mask = (input_sparsity is not None)
        if self.has_mask:
            self.register_buffer('input_mask', torch.tensor(input_sparsity, dtype=torch.float32))
        
        self.init_weights()
    
    def init_weights(self):
        # 初始化代码保持不变
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)
        with torch.no_grad():
            self.input_map.bias.data[self.hidden_size:2*self.hidden_size].fill_(1.0)
        nn.init.ones_(self.time_constants)
    
    def apply_mask_to_weights(self):
        # 掩码应用代码保持不变
        if self.has_mask:
            weight = self.input_map.weight.data
            try:
                expanded_mask = torch.zeros((4 * self.hidden_size, self.input_size), 
                                        device=weight.device, dtype=weight.dtype)
                for gate in range(4):
                    start_idx = gate * self.hidden_size
                    end_idx = (gate + 1) * self.hidden_size
                    for i in range(self.hidden_size):
                        if i < self.input_mask.shape[1]:
                            expanded_mask[start_idx + i, :] = self.input_mask[:, i]
                
                self.input_map.weight.data = weight * expanded_mask
            except Exception as e:
                print(f"警告: 掩码应用失败 {e}")
    
    def _compute_gate_values(self, input, h):
        """计算并返回门控值"""
        z = self.input_map(input) + self.recurrent_map(h)
        i, f, g, o = z.chunk(4, 1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        return i, f, g, o
    
    def _analytic_step(self, input, state, dt):
        """使用解析解方法求解"""
        h, c = state
        i, f, g, o = self._compute_gate_values(input, h)
        
        # 解析解形式
        tau = torch.abs(self.time_constants) + self._epsilon
        exp_factor = torch.exp(-dt / tau.unsqueeze(0))
        
        driving_term = f * c + i * g
        new_c = c * exp_factor + driving_term * (1.0 - exp_factor)
        new_h = o * torch.tanh(new_c)
        
        return new_h, new_c
    
    def _rk4_step(self, input, h, c, dt):
        """使用RK4方法积分一步"""
        # 预计算输入贡献
        input_contribution = self.input_map(input)
        
        def derivative(h_curr, c_curr):
            z = input_contribution + self.recurrent_map(h_curr)
            i, f, g, o = z.chunk(4, 1)
            
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            
            tau = torch.abs(self.time_constants) + self._epsilon
            dc = (f * c_curr + i * g - c_curr) / tau.unsqueeze(0)
            
            return dc
        
        # RK4积分步骤
        k1 = derivative(h, c)
        k2 = derivative(h, c + 0.5 * dt * k1)
        k3 = derivative(h, c + 0.5 * dt * k2)
        k4 = derivative(h, c + dt * k3)
        
        new_c = c + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 更新隐藏状态
        o = torch.sigmoid(self.input_map(input)[-self.hidden_size:] + 
                          self.recurrent_map(h)[-self.hidden_size:])
        new_h = o * torch.tanh(new_c)
        
        return new_h, new_c
    
    def _midpoint_step(self, input, h, c, dt):
        """使用中点法积分一步"""
        # 预计算输入贡献
        input_contribution = self.input_map(input)
        
        def derivative(h_curr, c_curr):
            z = input_contribution + self.recurrent_map(h_curr)
            i, f, g, o = z.chunk(4, 1)
            
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            
            tau = torch.abs(self.time_constants) + self._epsilon
            dc = (f * c_curr + i * g - c_curr) / tau.unsqueeze(0)
            
            return dc
        
        # 中点法积分
        k1 = derivative(h, c)
        c_mid = c + 0.5 * dt * k1
        k2 = derivative(h, c_mid)  # 使用h保持不变
        
        new_c = c + dt * k2
        
        # 更新隐藏状态
        o = torch.sigmoid(self.input_map(input)[-self.hidden_size:] + 
                          self.recurrent_map(h)[-self.hidden_size:])
        new_h = o * torch.tanh(new_c)
        
        return new_h, new_c
    
    def _euler_step(self, input, h, c, dt):
        """使用简单欧拉法积分一步"""
        i, f, g, o = self._compute_gate_values(input, h)
        
        tau = torch.abs(self.time_constants) + self._epsilon
        dc = (f * c + i * g - c) / tau.unsqueeze(0)
        new_c = c + dt * dc
        new_h = o * torch.tanh(new_c)
        
        return new_h, new_c
    
    def forward(self, input, state, elapsed_time=1.0):
        h, c = state
        
        if self.has_mask:
            self.apply_mask_to_weights()
        
        # 根据选择的方法进行积分
        if self._ode_method == 'analytic':
            return self._analytic_step(input, (h, c), elapsed_time)
        
        elif self._ode_method == 'rk4':
            dt = elapsed_time / self._ode_unfolds
            for _ in range(self._ode_unfolds):
                h, c = self._rk4_step(input, h, c, dt)
            return h, c
        
        elif self._ode_method == 'midpoint':
            dt = elapsed_time / self._ode_unfolds
            for _ in range(self._ode_unfolds):
                h, c = self._midpoint_step(input, h, c, dt)
            return h, c
        
        elif self._ode_method == 'mixed':
            # 使用混合策略：大步长用高阶方法，小步长用低阶方法
            coarse_steps = max(1, self._ode_unfolds // 3)
            dt_coarse = elapsed_time / coarse_steps
            
            # 大步长用解析解
            h, c = self._analytic_step(input, (h, c), dt_coarse)
            
            # 剩余时间用欧拉法细化
            remaining_time = elapsed_time - dt_coarse
            if remaining_time > 0:
                fine_steps = 3
                dt_fine = remaining_time / fine_steps
                for _ in range(fine_steps):
                    h, c = self._euler_step(input, h, c, dt_fine)
            
            return h, c
        
        else:  # 默认使用欧拉法
            dt = elapsed_time / self._ode_unfolds
            for _ in range(self._ode_unfolds):
                h, c = self._euler_step(input, h, c, dt)
            return h, c

class WiredLSTMODECell(nn.Module):
    def __init__(self, input_size, wiring, ode_unfolds=6, epsilon=1e-8):
        super(WiredLSTMODECell, self).__init__()
        
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'."
            )
        self._wiring = wiring
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        
        self._layers = []
        in_features = wiring.input_dim
        for l in range(wiring.num_layers):
            hidden_units = self._wiring.get_neurons_of_layer(l)

            if l == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                prev_layer_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_layer_neurons, :]
            cell = SparseLinearLSTMODECell(
                in_features,
                len(hidden_units),
                input_sparsity=input_sparsity,
                ode_unfolds=ode_unfolds,
                epsilon=epsilon
            )
            
            self.register_module(f"layer_{l}", cell)
            self._layers.append(cell)
            in_features = len(hidden_units)
    
    @property
    def state_size(self):
        return self._wiring.units
        
    @property
    def layer_sizes(self):
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]
    
    @property
    def num_layers(self):
        return self._wiring.num_layers
    
    @property
    def sensory_size(self):
        return self._wiring.input_dim
    
    @property
    def motor_size(self):
        return self._wiring.output_dim
    
    @property
    def output_size(self):
        return self.motor_size
    
    def forward(self, input, hx, elapsed_time=2.0):
        if not isinstance(hx, tuple) or len(hx) != 2:
            batch_size = input.size(0)
            device = input.device
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = torch.zeros((batch_size, self.state_size), device=device)
            hx = (h_state, c_state)
        
        h_state, c_state = hx
        h_states = list(torch.split(h_state, self.layer_sizes, dim=1))
        c_states = list(torch.split(c_state, self.layer_sizes, dim=1))
        
        new_h_states = []
        new_c_states = []
        inputs = input
        
        for i in range(self.num_layers):
            h, c = self._layers[i].forward(inputs, (h_states[i], c_states[i]), elapsed_time)
            inputs = h
            new_h_states.append(h)
            new_c_states.append(c)
        
        new_h_state = torch.cat(new_h_states, dim=1)
        new_c_state = torch.cat(new_c_states, dim=1)
        
        return h, (new_h_state, new_c_state)