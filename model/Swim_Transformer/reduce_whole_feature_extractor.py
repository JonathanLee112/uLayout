import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools

ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]
ENCODER_DENSENET = [
    'densenet121', 'densenet169', 'densenet161', 'densenet201'
]


def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )


'''
Encoder
'''
class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x);  features.append(x)  # 1/4
        x = self.encoder.layer2(x);  features.append(x)  # 1/8
        x = self.encoder.layer3(x);  features.append(x)  # 1/16
        x = self.encoder.layer4(x);  features.append(x)  # 1/32
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4


class Densenet(nn.Module):
    def __init__(self, backbone='densenet169', pretrained=True):
        super(Densenet, self).__init__()
        assert backbone in ENCODER_DENSENET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        self.final_relu = nn.ReLU(inplace=True)
        del self.encoder.classifier

    def forward(self, x):
        lst = []
        for m in self.encoder.features.children():
            x = m(x)
            lst.append(x)
        features = [lst[4], lst[6], lst[8], self.final_relu(lst[11])]
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.features.children()]
        block0 = lst[:4]
        block1 = lst[4:6]
        block2 = lst[6:8]
        block3 = lst[8:10]
        block4 = lst[10:]
        return block0, block1, block2, block3, block4


'''
Decoder
'''
class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''
    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)



class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c//2),
            ConvCompressH(in_c//2, in_c//2),
            ConvCompressH(in_c//2, in_c//4),
            ConvCompressH(in_c//4, out_c),
        )


    def forward(self, x, out_w):
        if x.size(2) * 2 == x.size(3):
            x = self.layer(x)
            assert out_w % x.shape[3] == 0
            factor = out_w // x.shape[3]
            x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
            x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
            x = x[..., factor:-factor]     
        return x


class GlobalHeightStage(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStage, self).__init__()
        self.cs = c1, c2, c3, c4
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConv(c1, c1//out_scale),
            GlobalHeightConv(c2, c2//out_scale),
            GlobalHeightConv(c3, c3//out_scale),
            GlobalHeightConv(c4, c4//out_scale),
        ])

    def forward(self, conv_list, out_w, idx):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
        ], dim=1)
        
        return feature

'''
Decoder pp module
'''
class GlobalHeightConvPP(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConvPP, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c//2),
            ConvCompressH(in_c//2, in_c//2),
            ConvCompressH(in_c//2, in_c//4),
            ConvCompressH(in_c//4, out_c),
        )

    def forward(self, x, out_w):
        x = self.layer(x)
        x = F.interpolate(x, size=(x.shape[2], 64), mode='bilinear', align_corners=False)    
        return x

class GlobalHeightStagePP(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStagePP, self).__init__()
        self.cs = c1, c2, c3, c4
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConvPP(c1, c1//out_scale),
            GlobalHeightConvPP(c2, c2//out_scale),
            GlobalHeightConvPP(c3, c3//out_scale),
            GlobalHeightConvPP(c4, c4//out_scale),
        ])

    def forward(self, conv_list, out_w, idx):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, 64)
            for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
        ], dim=1)
        return feature

'''
Reduce Feature Extractor
'''
class ReduceWholeFeatureExtractor(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone):
        super(ReduceWholeFeatureExtractor, self).__init__()
        self.backbone = backbone
        self.out_scale = 8
        self.step_cols = 4
        self.shift_window_size = 128
        self.shift_window_stride = self.shift_window_size // 2

        # Encoder
        if backbone.startswith('res'):
            self.feature_extractor = Resnet(backbone, pretrained=True)
        elif backbone.startswith('dense'):
            self.feature_extractor = Densenet(backbone, pretrained=True)
        else:
            raise NotImplementedError()

        # Inference channels number from each block of the encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 1024)
            c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(dummy)]
            c_last = (c1*8 + c2*4 + c3*2 + c4*1) // self.out_scale
        # Convert features from 4 blocks of the encoder into B x C x 1 x W'
        self.reduce_height_module = GlobalHeightStage(c1, c2, c3, c4, self.out_scale)
        self.reduce_height_module_pp = GlobalHeightStagePP(c1, c2, c3, c4, self.out_scale)
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False
        wrap_lr_pad(self)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std
    
    def _differenciate_perspective_panorama(self, x):
        
        '''
        input: 
            x, B x 3 x 512 x 1024
            contains panorama (full of color)
            contains perspective (color region is only (3,256,256) in random position other region is black)
        output:
            x_pano, B x 3 x 512 x 1024
            max x_pers, B x 3 x 384 x 384
            idx, B x 3
        '''
        x_pano = torch.zeros((0,3,512,1024)).to(x.device)
        x_pers = torch.zeros((0,3,512,256)).to(x.device)
        # differentiate perspective image or panorama image
        idx = torch.zeros((x.size(0),2)).int().to(x.device)
        
        for i in range(x.size(0)):
            # find color pixel
            color_u_idx = torch.nonzero(x[i, :].sum(dim=(0,1)) != 0).to(x.device)
            
            if len(color_u_idx) == 1024:
                x_pano = torch.cat((x_pano, x[i, :, :, :][None,...]), dim=0).to(x.device)
            else:
                color_v_idx = torch.nonzero(x[i, :].sum(dim=(0,2)) != 0).to(x.device)
                idx[i,0] = 1
                # h_avg = (min(color_u_idx)+max(color_u_idx)) // 2
                # if h_avg not in color_u_idx:
                #     if x.size(-1) - max(color_u_idx) > x_pers.size(-1) // 2:
                #         h_avg = (x.size(-1) + min(color_u_idx) + max(color_u_idx)) // 2
                #     else:
                #         h_avg = (-(x.size(-1) - max(color_u_idx)) + min(color_u_idx)) // 2
                # idx[i,1] = (h_avg - x.size(-1) // 2)
                # x[i] = torch.roll(x[i], tuple([-idx[i,1]]), dims=-1)
                x_pers = torch.cat((x_pers, x[i, :, :, ((x.size(-1)-x_pers.size(-1))//2):((x.size(-1)+x_pers.size(-1))//2)][None,...]),dim=0).to(x.device)

        return x_pano, x_pers, idx
    
    def _merge_feature_map(self, feature_pano, feature_pers, idx):
        '''
        input:
            feature_pano, B x C x W'
            feature_pers, B x C x W'
            idx, 2, B
        output:
            feature, B x C x W'
        '''
        feature = torch.zeros((idx.size(0), feature_pano.size(1), feature_pano.size(2))).to(feature_pano.device)
        if len(feature_pers) > 0:
            feature[idx[:,0] == 1, :, (128-32):(128+32)] = feature_pers

        if len(feature_pano) > 0:
            feature[idx[:,0] == 0] = feature_pano
        
        return feature
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, x):
        if x.shape[2] != 512 or x.shape[3] != 1024:
            raise NotImplementedError()
        x_pano, x_pers, idx = self._differenciate_perspective_panorama(x)
        if len(x_pers) > 0:
            x_pers = self._prepare_x(x_pers)
            conv_list_pers = self.feature_extractor(x_pers)
            feature_pers = self.reduce_height_module_pp(conv_list_pers, x.shape[3]//self.step_cols, idx)
        else:
            feature_pers = torch.zeros((0,1024,256)).to(x.device)
        if len(x_pano) > 0:
            x_pano = self._prepare_x(x_pano)
            conv_list_pano = self.feature_extractor(x_pano)
            feature_pano = self.reduce_height_module(conv_list_pano, x.shape[3]//self.step_cols, idx)
        else:
            feature_pano = torch.zeros((0,1024,256)).to(x.device)

        feature = self._merge_feature_map(feature_pano, feature_pers, idx)
        
        return feature
