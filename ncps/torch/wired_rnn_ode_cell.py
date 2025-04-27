import numpy as np
import torch
from torch import nn

class WiredRNNODECell(nn.Module):
    def __init__(
        self,
        input_size,
        wiring,
        nonlinearity="tanh",
        ode_unfolds=12,
        epsilon=1e-8,
    ):
        super(WiredRNNODECell, self).__init__()
        
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'."
            )
        self._wiring = wiring
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        
        if nonlinearity == "tanh":
            self.activation = torch.tanh
        elif nonlinearity == "relu":
            self.activation = torch.relu
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

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

            cell = SparseLinearRNNODECell(
                in_features,
                len(hidden_units),
                input_sparsity=input_sparsity,
                activation=self.activation,
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
    
    def forward(self, input, hx, elapsed_time=1.0):
        h_state = torch.split(hx, self.layer_sizes, dim=1)
        
        new_h_state = []
        inputs = input
        
        for i in range(self.num_layers):
            h = self._layers[i].forward(inputs, h_state[i], elapsed_time)
            inputs = h
            new_h_state.append(h)
        
        new_h_state = torch.cat(new_h_state, dim=1)
        
        return h, new_h_state

class SparseLinearRNNODECell(nn.Module):
    def __init__(self, input_size, hidden_size, input_sparsity=None, activation=torch.tanh, 
                 ode_unfolds=6, epsilon=1e-8, ode_solver='mixed'):
        super(SparseLinearRNNODECell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._ode_solver = ode_solver  # 'analytic', 'rk4', 'euler', 'mixed'

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        self.time_constants = nn.Parameter(torch.Tensor(hidden_size))
        
        if input_sparsity is not None:
            self.register_buffer('input_mask', torch.tensor(input_sparsity.T, dtype=torch.float32))
        else:
            self.input_mask = None
            
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
        nn.init.ones_(self.time_constants)
        
    def _get_masked_weight_ih(self):
        if self.input_mask is not None:
            return self.weight_ih * self.input_mask
        return self.weight_ih
    
    def _calculate_target_state(self, input, state):
        """计算目标状态 tanh(Wx + Uh + b)"""
        masked_weight_ih = self._get_masked_weight_ih()
        return self.activation(
            torch.mm(input, masked_weight_ih.t()) + self.bias_ih +
            torch.mm(state, self.weight_hh.t()) + self.bias_hh
        )
        
    def _analytic_solution(self, input, state, elapsed_time):
        """使用解析解计算结果"""
        target_state = self._calculate_target_state(input, state)
        tau = torch.abs(self.time_constants) + self._epsilon
        exp_factor = torch.exp(-elapsed_time / tau.unsqueeze(0))
        
        return state * exp_factor + target_state * (1.0 - exp_factor)
    
    def _euler_step(self, input, state, dt):
        """欧拉步进"""
        target_state = self._calculate_target_state(input, state)
        tau = torch.abs(self.time_constants) + self._epsilon
        dh = (target_state - state) / tau.unsqueeze(0)
        
        return state + dh * dt
    
    def _rk4_step(self, input, state, dt):
        """四阶Runge-Kutta步进"""
        masked_weight_ih = self._get_masked_weight_ih()
        input_proj = torch.mm(input, masked_weight_ih.t()) + self.bias_ih
        tau = torch.abs(self.time_constants) + self._epsilon
        
        # 定义ODE右端
        def f(s):
            recurrent = torch.mm(s, self.weight_hh.t()) + self.bias_hh
            target = self.activation(input_proj + recurrent)
            return (target - s) / tau.unsqueeze(0)
        
        k1 = f(state)
        k2 = f(state + dt*k1/2)
        k3 = f(state + dt*k2/2)
        k4 = f(state + dt*k3)
        
        return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    def forward(self, input, hx, elapsed_time=1.0):
        """前向传播，根据选择的求解器使用不同的算法"""
        if self._ode_solver == 'analytic':
            # 纯解析解，最快但可能不精确
            return self._analytic_solution(input, hx, elapsed_time)
            
        elif self._ode_solver == 'rk4':
            # RK4积分，精确但计算成本高
            steps = max(2, self._ode_unfolds // 2)  # 可以使用更少步数
            dt = elapsed_time / steps
            state = hx
            for _ in range(steps):
                state = self._rk4_step(input, state, dt)
            return state
            
        elif self._ode_solver == 'mixed':
            # 混合方法：解析解 + 微调
            # 先使用解析解获取一个好的初始近似
            state = self._analytic_solution(input, hx, elapsed_time * 0.9)
            
            # 然后用RK4进行1-2步微调
            dt = elapsed_time * 0.1 / 2
            for _ in range(2):
                state = self._rk4_step(input, state, dt)
            return state
            
        else:  # 默认使用优化的欧拉法
            # 使用缓存优化的欧拉法
            masked_weight_ih = self._get_masked_weight_ih()
            input_proj = torch.mm(input, masked_weight_ih.t()) + self.bias_ih
            tau = torch.abs(self.time_constants) + self._epsilon
            
            steps = self._ode_unfolds
            dt = elapsed_time / steps
            dt_tau = dt / tau.unsqueeze(0)
            
            state = hx
            for _ in range(steps):
                recurrent_proj = torch.mm(state, self.weight_hh.t()) + self.bias_hh
                target_state = self.activation(input_proj + recurrent_proj)
                state = state + (target_state - state) * dt_tau
                
            return state