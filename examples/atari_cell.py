import os
import sys
import torch
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import ncps.wirings.wirings
from ncps.torch.ltc import LTC
from ncps.wirings.wirings import NCP, FullyConnected, AutoNCP
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from ncps.torch.wired_rnn import WiredRNN
from ncps.torch.wired_wc_cell import WilsonCowanCell
class WilsonCowanConvBlock(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # 初始卷积分离不同通道的时序信息
        self.time_conv = nn.Conv2d(4, 16, 1)  # 1x1 卷积处理时序维度
        
        # 主要特征提取路径
        self.conv_path = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # 并行的动态变化特征提取路径 - 捕获帧间差异
        self.motion_path = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, groups=4),  # 分组卷积处理
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2),  # 输出通道较少，关注主要运动特征
        )
        
        # 特征融合
        self.fusion = nn.Conv2d(128 + 64, 128, 1)
        self.norm = nn.LayerNorm(128)  # 最终归一化
        
    def forward(self, x):
        # 时序信息处理
        x = self.time_conv(x)
        
        # 主特征路径
        main_features = self.conv_path(x)
        
        # 动态特征路径
        motion_features = self.motion_path(x)
        
        # 特征融合
        combined = torch.cat([main_features, motion_features], dim=1)
        x = self.fusion(combined)
        
        # 全局池化
        x = x.mean((-1, -2))
        x = self.norm(x)
        
        return x
class Conv3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean((-1, -2))
        return x
class Conv3RNNBase(nn.Module):
    def __init__(self, n_actions, hidden_size=64, cell_type='ltc', wiring_type='ncp', 
                 sparsity_level=0.5, use_ode=False):
        super().__init__()
        self.conv_block = Conv3Block()
        if wiring_type == 'fc':
            wiring = FullyConnected(hidden_size, output_dim=n_actions)
        else:
            wiring = AutoNCP(
                units=hidden_size,
                output_size=n_actions,
                sparsity_level=sparsity_level,
                seed=42
            )
        if cell_type == 'ltc':
            self.rnn = LTC(128, wiring, batch_first=True)
        else:
            self.rnn = WiredRNN(
                input_size=128,
                wiring=wiring,
                cell_type=cell_type,
                batch_first=True,
                return_sequences=True,
                use_ode=use_ode  # 新增参数传递
            )
    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.rnn(x, hx)
        return x, hx
class Conv3LTC_FC(Conv3RNNBase):
    def __init__(self, n_actions, hidden_size=64):
        super().__init__(n_actions, hidden_size, cell_type='ltc', wiring_type='fc')
class Conv3LTC_NCP(Conv3RNNBase):
    def __init__(self, n_actions, hidden_size=64, sparsity_level=0.5):
        super().__init__(n_actions, hidden_size, cell_type='ltc', wiring_type='ncp', sparsity_level=sparsity_level)
class Conv3CfC_FC(Conv3RNNBase):
    def __init__(self, n_actions, hidden_size=64):
        super().__init__(n_actions, hidden_size, cell_type='cfc', wiring_type='fc')
class Conv3CfC_NCP(Conv3RNNBase):
    def __init__(self, n_actions, hidden_size=64, sparsity_level=0.5):
        super().__init__(n_actions, hidden_size, cell_type='cfc', wiring_type='ncp', sparsity_level=sparsity_level)
class Conv3LSTM_FC(Conv3RNNBase):
    def __init__(self, n_actions, hidden_size=64, use_ode=False):
        super().__init__(n_actions, hidden_size, cell_type='lstm', wiring_type='fc', use_ode=use_ode)
class Conv3LSTM_NCP(Conv3RNNBase):
    def __init__(self, n_actions, hidden_size=64, sparsity_level=0.5, use_ode=False):
        super().__init__(n_actions, hidden_size, cell_type='lstm', wiring_type='ncp', 
                         sparsity_level=sparsity_level, use_ode=use_ode)
class Conv3RNN_FC(Conv3RNNBase):
    def __init__(self, n_actions, hidden_size=64, use_ode=False):
        super().__init__(n_actions, hidden_size, cell_type='rnn', wiring_type='fc', use_ode=use_ode)
class Conv3RNN_NCP(Conv3RNNBase):
    def __init__(self, n_actions, hidden_size=64, sparsity_level=0.5, use_ode=False):
        super().__init__(n_actions, hidden_size, cell_type='rnn', wiring_type='ncp', 
                         sparsity_level=sparsity_level, use_ode=use_ode)
class BreakoutWiring(ncps.wirings.Wiring):
    def __init__(self, seed=42, excitatory_ratio=0.8):
        super().__init__(19)
        self.set_output_dim(4)  # Breakout requires 4 output actions
        self._rng = np.random.RandomState(seed)
        self.excitatory_ratio = excitatory_ratio  # Ratio of excitatory neurons in biological systems
        self._layer1_neurons = list(range(0, 10))       # [0-9]
        self._layer2_neurons = list(range(10, 15))      # [10-14]
        self._layer3_neurons = list(range(15, 19))      # [15-18]
        self._layer1_to_layer2_conn = 3  # Each layer 1 neuron connects to 3 layer 2 neurons
        self._layer2_to_layer3_conn = 2  # Each layer 2 neuron connects to 2 layer 3 neurons
        self._layer3_to_layer2_conn = 3  # Each layer 3 neuron connects to 3 layer 2 neurons
        self._layer2_recurrent_conn = 2  # Each layer 2 neuron has 2 recurrent connections
        self._neuron_polarity = {}
        for neuron_id in self._layer1_neurons:
            self._neuron_polarity[neuron_id] = 1 if self._rng.random() < self.excitatory_ratio else -1
        for neuron_id in self._layer2_neurons:
            self._neuron_polarity[neuron_id] = 1 if self._rng.random() < 0.7 else -1
        for neuron_id in self._layer3_neurons:
            self._neuron_polarity[neuron_id] = 1 if self._rng.random() < 0.6 else -1
    @property
    def num_layers(self):
        return 3
    def get_neurons_of_layer(self, layer_id):
        if layer_id == 0:
            return self._layer3_neurons  # Note: output layer is the first layer
        elif layer_id == 1:
            return self._layer2_neurons
        elif layer_id == 2:
            return self._layer1_neurons
        raise ValueError(f"Unknown layer {layer_id}")
    def get_type_of_neuron(self, neuron_id):
        if neuron_id < 10:
            return "inter"     # Layer 1 contains interneurons
        elif neuron_id < 15:
            return "command"   # Layer 2 contains command neurons
        return "motor"        # Layer 3 contains motor neurons
    def build(self, input_shape):
        super().build(input_shape)
        for src in self._layer1_neurons:
            targets = self._rng.choice(
                self._layer2_neurons, 
                size=min(self._layer1_to_layer2_conn, len(self._layer2_neurons)), 
                replace=False
            )
            for dest in targets:
                polarity = self._neuron_polarity[src]
                self.add_synapse(src, dest, polarity)
        for src in self._layer2_neurons:
            targets = self._rng.choice(
                self._layer3_neurons, 
                size=min(self._layer2_to_layer3_conn, len(self._layer3_neurons)), 
                replace=False
            )
            for dest in targets:
                polarity = self._neuron_polarity[src]
                self.add_synapse(src, dest, polarity)
        for src in self._layer3_neurons:
            targets = self._rng.choice(
                self._layer2_neurons, 
                size=min(self._layer3_to_layer2_conn, len(self._layer2_neurons)), 
                replace=False
            )
            for dest in targets:
                polarity = self._neuron_polarity[src]
                if polarity > 0 and self._rng.random() < 0.4:  # 40% chance to flip excitatory to inhibitory
                    polarity = -polarity
                self.add_synapse(src, dest, polarity)  # Feedback connections have reduced strength
        for src in self._layer2_neurons:
            targets = self._rng.choice(
                self._layer2_neurons, 
                size=min(self._layer2_recurrent_conn, len(self._layer2_neurons)), 
                replace=False
            )
            for dest in targets:
                if src == dest:  # Self-connection
                    polarity = 1.0
                else:
                    polarity = self._neuron_polarity[src]
                    if polarity > 0 and self._rng.random() < 0.5:  # 50% chance for lateral inhibition
                        polarity = -1.0
                self.add_synapse(src, dest, polarity)
        for dest in self._layer1_neurons:
            sources = self._rng.choice(
                range(self.input_dim), 
                size=min(4, self.input_dim), 
                replace=False
            )
            for src in sources:
                polarity = 1 if self._rng.random() < 0.8 else -1
                self.add_sensory_synapse(src, dest, polarity)
class Conv3LTC_Breakout(nn.Module):
    def __init__(self, n_actions, hidden_size=64):
        super().__init__()
        self.conv_block = Conv3Block()
        self.wiring = BreakoutWiring()
        self.rnn = LTC(128, self.wiring, batch_first=True)
    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.rnn(x, hx)
        return x, hx
class Conv3WilsonCowan_FC(nn.Module):
    def __init__(self, n_actions, hidden_size=64, dt=0.1, use_rk4=True):
        super().__init__()
        # self.conv_block = Conv3Block()
        self.conv_block = WilsonCowanConvBlock()
        self.wc_cell = WilsonCowanCell(
            input_size=128,
            output_size=n_actions,
            hidden_size=hidden_size,
            connectivity='full',
            dt=dt,
            use_rk4=use_rk4
        )
    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.wc_cell(x, hx)
        return x, hx
class Conv3WilsonCowan_Random(nn.Module):
    def __init__(self, n_actions, hidden_size=64, sparsity_level=0.5, dt=0.1, use_rk4=True):
        super().__init__()
        # self.conv_block = Conv3Block()
        self.conv_block = WilsonCowanConvBlock()
        self.wc_cell = WilsonCowanCell(
            input_size=128,
            output_size=n_actions,
            hidden_size=hidden_size,
            connectivity='random',  # 'distance-dependent', 'hierarchical', 'small-world'
            sparsity=sparsity_level,
            dt=dt,
            use_rk4=use_rk4
        )
    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.wc_cell(x, hx)
        return x, hx
class Conv3WilsonCowan_Modular(nn.Module):
    def __init__(self, n_actions, hidden_size=64, sparsity_level=0.5, dt=0.1, use_rk4=True):
        super().__init__()
        # self.conv_block = Conv3Block()
        self.conv_block = WilsonCowanConvBlock()
        self.wc_cell = WilsonCowanCell(
            input_size=128,
            output_size=n_actions,
            hidden_size=hidden_size,
            connectivity='modular',  # 使用分层连接结构
            sparsity=sparsity_level,
            dt=dt,
            use_rk4=use_rk4
        )
        
    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.wc_cell(x, hx)
        return x, hx
class Conv3WilsonCowan_SmallWorld(nn.Module):
    def __init__(self, n_actions, hidden_size=64, sparsity_level=0.5, dt=0.1, use_rk4=True):
        super().__init__()
        # self.conv_block = Conv3Block()
        self.conv_block = WilsonCowanConvBlock()
        self.wc_cell = WilsonCowanCell(
            input_size=128,
            output_size=n_actions,
            hidden_size=hidden_size,
            connectivity='small-world',  # 使用分层连接结构
            sparsity=sparsity_level,
            dt=dt,
            use_rk4=use_rk4
        )
        
    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.wc_cell(x, hx)
        return x, hx
class Conv3WilsonCowan_Hierarchical(nn.Module):
    def __init__(self, n_actions, hidden_size=64, sparsity_level=0.5, dt=0.1, use_rk4=True):
        super().__init__()
        # self.conv_block = Conv3Block()
        self.conv_block = WilsonCowanConvBlock()
        self.wc_cell = WilsonCowanCell(
            input_size=128,
            output_size=n_actions,
            hidden_size=hidden_size,
            connectivity='hierarchical',  # 使用分层连接结构
            sparsity=sparsity_level,
            dt=dt,
            use_rk4=use_rk4
        )
        
    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.wc_cell(x, hx)
        return x, hx
class Conv3WilsonCowan_DistanceDependent(nn.Module):
    def __init__(self, n_actions, hidden_size=64, sparsity_level=0.5, dt=0.1, use_rk4=True):
        super().__init__()
        self.conv_block = Conv3Block()
        # self.conv_block = WilsonCowanConvBlock()
        self.wc_cell = WilsonCowanCell(
            input_size=128,
            output_size=n_actions,
            hidden_size=hidden_size,
            connectivity='distance-dependent',  # 使用分层连接结构
            sparsity=sparsity_level,
            dt=dt,
            use_rk4=use_rk4
        )
        
    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.wc_cell(x, hx)
        return x, hx
