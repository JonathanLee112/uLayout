import torch.nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb

from model.Swim_Transformer.reduce_whole_feature_extractor import ReduceWholeFeatureExtractor
from model.Swim_Transformer.modules.swg_transformer import SWG_Transformer

class LGT_Net(nn.Module):
    def __init__(self, backbone='resnet50', dropout=0.0, output_name='Horizon',
                 decoder_name='SWG_Transformer', win_size=8, depth=6,
                 ape=None, rpe=None, corner_heat_map=False, rpe_pos=1):
        super(LGT_Net, self).__init__()

        self.patch_num = 256
        # self.patch_num = 64
        # self.patch_num = 128
        self.patch_dim = 1024
        self.decoder_name = decoder_name
        self.output_name = output_name
        self.corner_heat_map = corner_heat_map
        self.dropout_d = dropout
        self.step_cols = 4

        
        # feature extractor
        self.feature_extractor = ReduceWholeFeatureExtractor(backbone)


        if 'Transformer' in self.decoder_name:
            # transformer encoder
            transformer_dim = self.patch_dim
            transformer_layers = depth
            transformer_heads = 8
            transformer_head_dim = transformer_dim // transformer_heads
            transformer_ff_dim = 2048
            rpe = None if rpe == 'None' else rpe
            
            self.transformer = SWG_Transformer(dim=transformer_dim, depth=transformer_layers,
                                            heads=transformer_heads, dim_head=transformer_head_dim,
                                            mlp_dim=transformer_ff_dim, win_size=win_size,
                                            dropout=self.dropout_d, patch_num=self.patch_num,
                                            ape=ape, rpe=rpe, rpe_pos=rpe_pos)
        elif self.decoder_name == 'LSTM':
            self.bi_rnn = nn.LSTM(input_size=self.feature_extractor.c_last,
                                  hidden_size=self.patch_dim // 2,
                                  num_layers=2,
                                  dropout=self.dropout_d,
                                  batch_first=False,
                                  bidirectional=True)
            self.drop_out = nn.Dropout(self.dropout_d)
        else:
            raise NotImplementedError("Only support *Transformer and LSTM")

        if self.output_name == 'Horizon':
            # horizon output
            self.linear = nn.Linear(in_features=self.patch_dim,
                                    out_features=3 * self.step_cols)
            self.linear.bias.data[0*self.step_cols:1*self.step_cols].fill_(-1)
            self.linear.bias.data[1*self.step_cols:2*self.step_cols].fill_(-0.478)
            self.linear.bias.data[2*self.step_cols:3*self.step_cols].fill_(0.425)

        else:
            raise NotImplementedError("Unknown output")

        if self.corner_heat_map:
            # corners heat map output
            self.linear_corner_heat_map_output = nn.Linear(in_features=self.patch_dim, out_features=1)

        self.name = f"{self.decoder_name}_{self.output_name}_Net"

    def horizon_output(self, x):
        """
        :param x: [ b, 256(patch_num), 1024(d)]
        :return: {
            'floor_boundary':  [b, 256(patch_num)]
            'ceil_boundary': [b, 256(patch_num)]
        }
        """
        x = x.permute(1, 0, 2)  # [w, b, c*h]
        x = self.linear(x)
        x = x.view(x.shape[0], x.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
        x = x.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
        output = x.contiguous().view(x.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
        cor = output[:, :1]  # [b, 1, seq_len*step_cols]
        bon = output[:, 1:]

        ######### for training both panorama and perspective or solely training panorama, apply sigmoid to bon ###########
        ######### for solely training perspective, DO NOT apply sigmoid to bon and directly train not evaluate first ###########
        
        bon = torch.sigmoid(bon)
        bon_0 = bon[:, 0:1, :] * -math.pi / 2
        bon_1 = bon[:, 1:, :] * math.pi / 2
        bon = torch.cat((bon_0, bon_1), dim=1)

        return bon, cor

    def forward(self, x):
        """
        :param x: [b, 3(d), 512(h), 1024(w)]
        :return: {
            'depth': [b, 256(patch_num & d)]
            'ratio': [b, 1(d)]
        }
        """
        # feature extractor
        # HorizonNetFeatureExtractor
        x = self.feature_extractor(x)  # [b 1024(d) 256(w)]
        
        if 'Transformer' in self.decoder_name:
            # transformer decoder
            x = x.permute(0, 2, 1)  # [b 256(patch_num) 1024(d)]
            x = self.transformer(x)  # [b 256(patch_num) 1024(d)]
        elif self.decoder_name == 'LSTM':
            # lstm decoder
            x = x.permute(2, 0, 1)  # [256(patch_num), b, 1024(d)]
            self.bi_rnn.flatten_parameters()
            x, _ = self.bi_rnn(x)  # [256(patch_num & seq_len), b, 1024(d)]
            x = x.permute(1, 0, 2)  # [b, 256(patch_num), 1024(d)]
            x = self.drop_out(x)

        bon = None
        cor = None

        bon, cor = self.horizon_output(x)

        return bon, cor
        
