#!/usr/bin/env python

""" Define ASFormer arch for video classification in pytorch
"""
from __future__ import print_function
import sys
import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models import layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p * idx_decoder)


class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_size, output_size, channel_masking_rate, att_type,
                 alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_size, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList(
            [layers.AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in
             range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, output_size, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        if mask is not None:
            out = self.conv_out(feature) * mask[:, 0:1, :]
        else:
            out = self.conv_out(feature)

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_size, output_size, att_type, alpha):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_size, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [layers.AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in
             # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, output_size, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        if mask is not None:
            out = self.conv_out(feature) * mask[:, 0:1, :]
        else:
            out = self.conv_out(feature)

        return out, feature


class ASFormer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_size, output_size, channel_masking_rate):
        super(ASFormer, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_size, output_size, channel_masking_rate,
                               att_type='sliding_att', alpha=1)
        self.decoders = nn.ModuleList([copy.deepcopy(
            Decoder(num_layers, r1, r2, num_f_maps, output_size, output_size, att_type='sliding_att',
                    alpha=exponential_descrease(s))) for s in range(num_decoders)])  # num_decoders

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        out, feature = self.encoder(x, mask)
        outputs = [out.permute(0, 2, 1)]

        for decoder in self.decoders:
            if mask is not None:
                out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask)
            else:
                out, feature = decoder(F.softmax(out, dim=1), feature, mask)
            outputs += [out.permute(0, 2, 1)]

        return outputs
