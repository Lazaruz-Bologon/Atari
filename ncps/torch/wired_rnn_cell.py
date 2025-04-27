import numpy as np
import torch
from torch import nn

class WiredRNNCell(nn.Module):
    def __init__(
        self,
        input_size,
        wiring,
        nonlinearity="tanh",
    ):
        super(WiredRNNCell, self).__init__()
        
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'."
            )
        self._wiring = wiring
        
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

            cell = SparseLinearRNNCell(
                in_features,
                len(hidden_units),
                input_sparsity=input_sparsity,
                activation=self.activation
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
        h_state = torch.split(hx, self.layer_sizes, dim=1)
        
        new_h_state = []
        inputs = input
        
        for i in range(self.num_layers):
            h = self._layers[i].forward(inputs, h_state[i])
            inputs = h
            new_h_state.append(h)
        
        new_h_state = torch.cat(new_h_state, dim=1)
        
        return h, new_h_state

class SparseLinearRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, input_sparsity=None, activation=torch.tanh):
        super(SparseLinearRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))

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
    
    def forward(self, input, hx):
        if self.input_mask is not None:
            masked_weight_ih = self.weight_ih * self.input_mask
        else:
            masked_weight_ih = self.weight_ih
        h = self.activation(
            torch.mm(input, masked_weight_ih.t()) + self.bias_ih +
            torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        )
        
        return h