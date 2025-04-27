import numpy as np
import torch
from torch import nn

class SparseLinearLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, input_sparsity=None):
        super(SparseLinearLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.has_mask = (input_sparsity is not None)
        if self.has_mask:
            self.register_buffer('input_mask', torch.tensor(input_sparsity, dtype=torch.float32))
        
        self.init_weights()
    
    def init_weights(self):
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
    
    def apply_mask_to_weights(self):
        if self.has_mask:
            weight = self.input_map.weight.data
            
            try:
                expanded_mask = torch.zeros((4 * self.hidden_size, self.input_size), 
                                        device=weight.device, dtype=weight.dtype)
                for gate in range(4):
                    start_idx = gate * self.hidden_size
                    end_idx = (gate + 1) * self.hidden_size
                    for i in range(self.hidden_size):
                        if i < self.input_mask.shape[1]:  # 确保索引在范围内
                            expanded_mask[start_idx + i, :] = self.input_mask[:, i]
                
                self.input_map.weight.data = weight * expanded_mask
            except Exception as e:
                print(f"警告: 掩码应用失败 {e}")
    
    def forward(self, input, state):
        h, c = state

        if self.has_mask:
            self.apply_mask_to_weights()

        z = self.input_map(input) + self.recurrent_map(h)
        i, f, g, o = z.chunk(4, 1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        
        return h, c


class WiredLSTMCell(nn.Module):
    def __init__(self, input_size, wiring):
        super(WiredLSTMCell, self).__init__()
        
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'."
            )
        self._wiring = wiring
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
            cell = SparseLinearLSTMCell(
                in_features,
                len(hidden_units),
                input_sparsity=input_sparsity
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
    
    def forward(self, input, hx):
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
            h, c = self._layers[i].forward(inputs, (h_states[i], c_states[i]))
            inputs = h
            new_h_states.append(h)
            new_c_states.append(c)
        
        new_h_state = torch.cat(new_h_states, dim=1)
        new_c_state = torch.cat(new_c_states, dim=1)
        
        return h, (new_h_state, new_c_state)