from typing import List

import numpy as np
import pyquaternion
from mmengine.model import BaseModule
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mmdet3d.registry import MODELS
from ..utils.lidar2global import output_to_nusc_box
from torch.nn import GRU

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

@MODELS.register_module()
class FeatureFusion(BaseModule):

    def __init__(self, in_channel, out_channel, hidden_channel):
        """

        Args:
            in_channel:
            out_channel:
            hidden_channel:
        """
        super(FeatureFusion, self).__init__()
        self.mlp1 = MLP(in_channel, hidden_channel, out_channel).cuda()
        self.mlp2 = MLP(out_channel, hidden_channel, out_channel).cuda()

    def forward(self, x, y):
        """

        Args:
            x: box feature [B, N, 12] or [N, 12]
            y: tracking embedding feature [B, N, 256] or [N, 256]

        Returns:
            fusion_feature
        """
        if len(x.shape) == 3:
            l, n, d1 = x.shape
            l, n, d2 = y.shape

            x = x.reshape(-1, d1)
            y = y.reshape(-1, d2)

            x = self.mlp1(x)

            y = x + y
            # y = x

            fusion_feature = self.mlp2(y)

            fusion_feature = fusion_feature.reshape(l, n, -1)
        else:
            x = self.mlp1(x)
            # y = x
            y = x + y

            fusion_feature = self.mlp2(y)

        return fusion_feature


@MODELS.register_module()
class RNN(BaseModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.intput_embed = MLP(input_size, hidden_size, hidden_size)
        self.output_embed = MLP(hidden_size, hidden_size, output_size)

    def forward(self, x: torch.Tensor, lengths=None):
        """
        gru method
        Args:
            x: [B, L, D]
            lengths: [B]

        Returns:
            out: [B, L, D]
        """
        # forward pass

        # inference
        if lengths is not None:
            self.eval()
            x = self.intput_embed(x)
            # 变长序列输入到RNN之前必须pack
            padded_tk_tracks = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

            out, h = self.rnn(padded_tk_tracks)
            packed_tk_tracks = pad_packed_sequence(out, batch_first=True)[0]

            out = self.output_embed(packed_tk_tracks)
            n, l, d = out.shape
            out_indexes = [i - 1 for i in lengths]
            out_indexes = torch.tensor(out_indexes, dtype=torch.int64)
            out = out[torch.arange(n), out_indexes, :]


        else:
            x = self.intput_embed(x)
            out, h = self.rnn(x)
            out = self.output_embed(out)

        return out























