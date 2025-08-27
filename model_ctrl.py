#!/usr/bin/env python
# encoding: utf-8

from typing import Literal

import torch
import torch.nn as nn
from typing import Tuple, List, Union, Optional


class CTRL(nn.Module):
    """
        Collaborative Temporal Representation Learning (CTRL).
        Author: Zhijin Wang, Email: zhijinecnu@gmail.com

        Y Hu, S Wu, Y Chen, X He, Z Xie, Z Wang, X Liu, Y Fu.
        CTRL: Collaborative Temporal Representation Learning for Wind Power Forecasting.
        ACM EITCE 2024, doi: 10.1145/3711129.371133

        :param input_window_size: input window size.
        :param input_vars:  number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param rnn_hidden_size: hidden size of RNN.
        :param rnn_num_layers: number of layers of RNN.
        :param rnn_bidirectional: if True, use bidirectional RNN.
        :param activation: 'str' type, activation function.
        :param use_instance_scale: if True, use instance scale (RevIN).
        :param dropout_rate: dropout rate.
    """

    def __init__(self, input_window_size: int, input_vars: int = 1, output_window_size: int = 1, output_vars: int = 1,
                 rnn_hidden_size: int = 32, rnn_num_layers: int = 2, rnn_bidirectional: bool = False,
                 activation: str = 'linear', use_instance_scale: bool = True, dropout_rate: float = 0.):
        super(CTRL, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bidirectional = rnn_bidirectional
        self.activation = activation
        self.use_instance_scale = use_instance_scale

        # RNN
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bidirectional = rnn_bidirectional

        self.rnn = nn.GRU(input_size=self.input_vars, hidden_size=self.rnn_hidden_size,
                          num_layers=self.rnn_num_layers, bidirectional=self.rnn_bidirectional,
                          batch_first=True, dropout=dropout_rate)

        in_features = self.rnn_hidden_size * 2 if self.rnn_bidirectional else self.rnn_hidden_size
        self.l1 = nn.Linear(in_features, self.input_vars)

        # COAT
        self.dr0 = DirectionalRepresentation(self.input_window_size, self.input_vars, 0, activation, dropout_rate)
        self.dr1 = DirectionalRepresentation(self.input_window_size, self.input_vars, 1, activation, dropout_rate)
        self.dr2 = DirectionalRepresentation(self.input_window_size, self.input_vars, 2, activation, dropout_rate)

        self.combination_size = self.input_vars * 4
        self.ar = GAR(self.input_window_size, self.output_window_size)

        self.fc = MLP(self.combination_size, [self.combination_size, self.combination_size // 2], input_vars,
                      None, activation, dropout_rate)

        self.fc1 = MLP(self.input_vars, [self.input_vars, self.input_vars // 2], output_vars,
                       None, activation, dropout_rate)

        if self.use_instance_scale:
            self.inst_scaler = InstanceStandardScale(self.input_vars, 1e-5)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            x -> (batch_size, input_window_size, input_vars)
         """

        if self.use_instance_scale:
            x = self.inst_scaler.fit_transform(x, x_mask)

        if x_mask is not None:
            x[~x_mask] = 0.0  # set the masked values to zero

        # rnn as encoder
        rnn_out, _ = self.rnn(x)  # -> [batch_size, input_window_size, hidden_size]
        rnn_out = self.l1(rnn_out)  # -> [batch_size, input_window_size, input_vars]

        # coat as linear mappings: rnn > DR
        out0 = self.dr0(rnn_out)  # => [batch_size, input_window_size, input_vars]
        out1 = self.dr1(rnn_out)  # => [batch_size, input_window_size, input_vars]
        out2 = self.dr2(rnn_out)  # => [batch_size, input_window_size, input_vars]

        out = torch.cat([out0, out1, out2, x], dim=2)  # => [batch_size, input_window_size, comb_size]

        # Linearly mapping to outputs
        out = self.ar(out)  # => [batch_size, output_window_size, comb_size]
        out = self.fc(out)  # => [batch_size, output_window_size, input_vars]
        if self.use_instance_scale:
            out = self.inst_scaler.inverse_transform(out)

        if self.input_vars != self.output_vars:
            out = self.fc1(out)  # => [batch_size, output_window_size, output_vars]

        return out


class GAR(nn.Module):
    """
         Global autoregression.
         Note: the target dimension is equal to the input dimension, input_vars === output_vars.
        :param input_window_size: input window size.
        :param output_window_size: output window size.
        :param bias: if True, adds a learnable bias to the output.
        :param activation: type str, the activation function.
    """

    def __init__(self, input_window_size: int, output_window_size: int = 1, bias: bool = True,
                 activation: str = 'linear'):
        super(GAR, self).__init__()
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.activation = activation

        self.l1 = nn.Linear(self.input_window_size, self.output_window_size, bias)
        self.activate = get_activation_cls(activation)()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            :param x: input tensor, the shape is ``(..., input_window_size, input_vars)``.
            :param x_mask: mask tensor of input tensor, the shape is ``(..., input_window_size, input_vars)``.
            :return: output tensor, the shape is ``(..., output_window_size, input_vars)``.
        """

        if x_mask is not None:
            x[~x_mask] = 0.

        x = x.transpose(-1, -2)  # -> (..., input_vars, input_window_size)
        x = self.l1(x)  # -> (..., input_vars, output_window_size)
        x = x.transpose(-1, -2)  # x -> (..., output_window_size, input_vars)

        x = self.activate(x)

        return x


def get_activation_cls(activation: str = 'linear'):
    """
    Get the activation class.

    :param activation: The activation function to use.
    :param params: inplace = True or False.

    :return: The activation function.
    """
    activation_dict = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'linear': nn.Identity,
    }

    activation_fn = activation_dict.get(activation, activation_dict['linear'])
    return activation_fn


class MLP(nn.Module):
    """
        Multiple Layer Perceptron (MLP).

        :param input_dim: input feature dimension.
        :param hidden_units: list of positive integer, the layer number and units in each layer.
        :param output_dim: output feature dimension.
        :param layer_norm: layer normalization in ('DyT' or 'LN'). If None, no layer normalization is applied.
        :param activation: the activation function to use.
        :param dropout_rate: float in [0, 1). Fraction of the units to dropout.
    """

    def __init__(self, input_dim: int, hidden_units: Union[Tuple[int, ...], List[int]], output_dim: int = 1,
                 layer_norm: Literal['DyT', 'LN'] = None, activation: str = None, dropout_rate: float = 0.):
        super(MLP, self).__init__()
        self.input_dim = input_dim  # input feature dimension
        self.hidden_units = hidden_units  # hidden units for each layer
        self.output_dim = output_dim  # output feature dimension
        self.layer_norm = layer_norm
        self.activation = activation  # name of the activation function
        self.dropout_rate = dropout_rate

        units = [self.input_dim, *self.hidden_units, self.output_dim]

        self.mlp = nn.Sequential()
        for i in range(len(units) - 1):
            self.mlp.add_module(f'layer_{i}', nn.Linear(units[i], units[i + 1]))

            if i < len(units) - 2:
                if self.layer_norm is not None:
                    if self.layer_norm == 'DyT':
                        self.mlp.add_module(f'dynamic_tanh_{i}', DynamicTanh(units[i + 1]))
                    elif self.layer_norm == 'LN':
                        self.mlp.add_module(f'layer_norm_{i}', nn.LayerNorm(units[i + 1]))
                if self.activation is not None or self.activation != 'linear':
                    self.mlp.add_module(f'activation_{i}', get_activation_cls(self.activation)())
                if self.dropout_rate > 0:
                    self.mlp.add_module(f'dropout_{i}', nn.Dropout(p=self.dropout_rate))

    def forward(self, x: torch.Tensor):
        """
            x, the input tensor, shape: ``(batch_size, input_dim)``
        """
        out = self.mlp(x)  # shape: (batch_size, output_dim)
        return out


class DynamicTanh(nn.Module):
    """
        Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu,
        Transformers without Normalization, 2025
        https://arxiv.org/abs/2503.10622

        Dynamic Tanh layer norm.

        :param dim: int, the dimension of the input tensor.
        :param init_alpha: float, the initial value of alpha.
    """

    def __init__(self, dim: int, init_alpha: float = 1.):
        super(DynamicTanh, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
            :param x: torch.Tensor, input tensor of shape (batch_size, seq_len, dim).
        """

        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta


class InstanceStandardScale():
    """
        Standard normalization on batch instances in forward feedback. A.k.a. **ReVIN**.

        Kim T, Kim J, Tae Y, et al.
        Reversible instance normalization for accurate time-series forecasting against distribution shift
        ICLR 2021.

        :param num_features: if num_features > 0, then use affine parameters, else not.
        :param epsilon: the default value is 1e-5.
    """

    def __init__(self, num_features: int = -1, epsilon: float = 1e-5):
        self.num_features = num_features
        self.epsilon = epsilon

        self.miu = None
        self.sigma = None

        if self.num_features > 0:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def fit(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        """
            Fit the instance normalization parameters.
            :param x: the input tensor, shape is [..., num_features].
            :param x_mask: the mask tensor, shape is [..., num_features].
            :return: self, the fitted scaler.
        """

        assert x.ndim > 2, "The input tensor must be at least 3D."

        dim2reduce = tuple(range(1, x.ndim - 1))

        if x_mask is not None:
            assert x.shape == x_mask.shape, 'x and mask must have the same shape.'
            nan_mask = ~x_mask
            valid_counts = (~nan_mask).sum(dim=dim2reduce, keepdim=True).float()

            x_copy = x.clone()
            x_copy[nan_mask] = 0

            self.miu = x_copy.sum(dim=dim2reduce, keepdim=True) / (
                    valid_counts + self.epsilon)  # avoid division by zero
            self.sigma = (x_copy.var(dim=dim2reduce, unbiased=False, keepdim=True) + self.epsilon).sqrt()

            if self.num_features > 0:
                self.affine_weight.data.fill_(1.0)
                self.affine_bias.data.fill_(0.0)

            return self

        self.miu = x.mean(dim=dim2reduce, keepdim=True).detach()
        self.sigma = (x.var(dim=dim2reduce, keepdim=True, unbiased=False) + self.epsilon).sqrt().detach()
        return self

    def transform(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        """ Normalization """
        t = (x - self.miu) / self.sigma  # -> [..., num_features]
        if self.num_features > 0:
            t = (t * self.affine_weight) + self.affine_bias
        return t

    def fit_transform(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        self.fit(x, x_mask)
        normalized_x = self.transform(x, x_mask)
        return normalized_x

    def inverse_transform(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        """ De-normalize.  """
        if self.num_features > 0:
            x = (x - self.affine_bias) / (self.affine_weight + self.epsilon ** 2)
        inv = x * self.sigma + self.miu  # -> [..., num_features]
        return inv


class DirectionalRepresentation(nn.Module):
    """
        Learning directional representation from a windowed time series.
        Element-wise attention mechanism. (Good at short-sequence)
     """

    def __init__(self, window_size: int, input_size: int, r_dim: int = 0,
                 activation: str = 'linear', dropout_rate: float = 0.):
        super(DirectionalRepresentation, self).__init__()

        self.window_size = window_size
        self.input_size = input_size
        self.r_dim = r_dim
        self.activation = get_activation_cls(activation)()  # default as linear

        self.weight = nn.Parameter(torch.zeros(self.window_size, self.input_size))
        nn.init.uniform_(self.weight, a=0.01, b=0.1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
            :param x: [batch_size, window_size, input_size]
        """
        x = self.activation(x)
        out = x * self.weight  # element-wise linear mapping

        # out = self.activation(out)

        if self.r_dim != -1:
            out = out.softmax(dim=self.r_dim)

        # out = out * x   # element-wise attention, useful for short-sequence: infectious disease cases

        out = self.dropout(out)

        return out
