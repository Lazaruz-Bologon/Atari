import torch
from torch import nn
import numpy as np
from typing import Optional, Union
import ncps
from .wired_rnn_cell import WiredRNNCell
from .wired_lstm_cell import WiredLSTMCell
from .wired_cfc_cell import WiredCfCCell
from .wired_rnn_ode_cell import WiredRNNODECell
from .wired_lstm_ode_cell import WiredLSTMODECell
class WiredRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        wiring,
        cell_type="rnn",
        return_sequences: bool = True,
        batch_first: bool = True,
        nonlinearity="tanh",
        use_ode: bool = False,
        ode_unfolds: int = 6,
    ):
        """支持Wiring连接的RNN

        :param input_size: 输入特征数量
        :param wiring: 连接结构 (ncps.wirings.Wiring 实例)
        :param cell_type: RNN细胞类型，"rnn", "lstm", "cfc" 之一
        :param return_sequences: 是否返回完整序列还是仅最后一个输出
        :param batch_first: 批处理维度是否为第一维
        :param nonlinearity: 仅用于RNN细胞，"tanh"或"relu"
        :param use_ode: 是否使用ODE求解模式
        :param ode_unfolds: ODE求解的步数
        """
        super(WiredRNN, self).__init__()
        self.input_size = input_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.cell_type = cell_type.lower()
        self.use_ode = use_ode
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'."
            )
        self._wiring = wiring
        if self.cell_type == "rnn":
            if use_ode:
                self.rnn_cell = WiredRNNODECell(
                    input_size,
                    wiring,
                    nonlinearity=nonlinearity,
                    ode_unfolds=ode_unfolds
                )
            else:
                self.rnn_cell = WiredRNNCell(
                    input_size,
                    wiring,
                    nonlinearity=nonlinearity
                )
        elif self.cell_type == "lstm":
            if use_ode:
                self.rnn_cell = WiredLSTMODECell(
                    input_size,
                    wiring,
                    ode_unfolds=ode_unfolds
                )
            else:
                self.rnn_cell = WiredLSTMCell(
                    input_size,
                    wiring
                )
        elif self.cell_type == "cfc":
            self.rnn_cell = WiredCfCCell(
                input_size,
                wiring,
                mode="default"
            )
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
    
    @property
    def state_size(self):
        return self._wiring.units
    
    @property
    def output_size(self):
        return self._wiring.output_dim
    
    def forward(self, input, hx=None, timespans=None):
        """前向传播

        :param input: 输入张量，形状为 (L,C) 无批次模式，或 (B,L,C) 当 batch_first=True，或 (L,B,C) 当 batch_first=False
        :param hx: RNN的初始隐藏状态
        :param timespans: 用于CfC的时间跨度
        :return: 一对 (output, hx)，其中output是输出，hx是最终隐藏状态
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)
        
        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)
        if hx is None:
            if self.cell_type == "lstm":
                h_state = torch.zeros((batch_size, self.state_size), device=device)
                c_state = torch.zeros((batch_size, self.state_size), device=device)
                hx = (h_state, c_state)
            else:
                hx = torch.zeros((batch_size, self.state_size), device=device)
        
        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()
            
            if self.cell_type == "cfc" or (self.use_ode and self.cell_type in ["rnn", "lstm"]):
                h_out, hx = self.rnn_cell(inputs, hx, ts)
            else:
                h_out, hx = self.rnn_cell(inputs, hx)
            if self.output_size < self.state_size:
                try:
                    if hasattr(self._wiring, '_motor_neurons'):
                        motor_neurons = self._wiring._motor_neurons
                    else:
                        motor_neurons = list(range(self._wiring.units - self.output_size, self._wiring.units))
                    valid_indices = [idx for idx in motor_neurons if idx < h_out.size(1)]
                    if len(valid_indices) >= self.output_size:
                        valid_indices = valid_indices[:self.output_size]
                    else:
                        valid_indices = list(range(max(0, h_out.size(1) - self.output_size), h_out.size(1)))
                    h_out = h_out[:, valid_indices]
                except Exception as e:
                    print(f"警告：提取电机神经元输出时出错: {e}")
                    h_out = h_out[:, -self.output_size:]
            
            if self.return_sequences:
                output_sequence.append(h_out)
        
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = h_out
        
        if not is_batched:
            readout = readout.squeeze(batch_dim)
            if self.cell_type == "lstm":
                hx = (hx[0].squeeze(0), hx[1].squeeze(0))
            else:
                hx = hx.squeeze(0)
        
        return readout, hx