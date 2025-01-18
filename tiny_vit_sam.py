# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath as TimmDropPath,\
    to_2tuple, trunc_normal_
from timm.models.registry import register_model
from typing import Tuple
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(drop_prob={self.drop_prob})'
        return msg
class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * \
            self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)
#残差结构的卷积层
class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x
#瓶颈结构卷积，并下采样
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c=2
        if(out_dim==320 or out_dim==448 or out_dim==576):   #这些都是最后一层的特征数，那么就不用下采样了
            stride_c=1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x
class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth,
                 activation,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 conv_expand_ratio=4.,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path,
                   )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None]*x + self.bias[:, None, None]
        return x

class DCFormer(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=2,
                 embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
                 num_heads=[2, 4, 5, 10],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.0,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 ):
        super().__init__()
        self.img_size=img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(dim=embed_dims[i_layer],
                        input_resolution=(patches_resolution[0] // (2 ** (i_layer-1 if i_layer == 3 else i_layer)),
                                patches_resolution[1] // (2 ** (i_layer-1 if i_layer == 3 else i_layer))),
                        #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                        #                     patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=PatchMerging if (
                              i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = DCFormerLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)

        # Classifier head
        self.avg_pool = nn.AvgPool2d(kernel_size=16)
        self.norm_head = nn.LayerNorm(256)
        #self.norm_head = nn.BatchNorm1d(256)
        self.head = nn.Linear(256, self.num_classes) if self.num_classes > 0 else torch.nn.Identity()

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )
    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
        #print("LR SCALES:", lr_scales)

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward_features(self, x):
        # x: (N, 3, 256, 256)
        x = self.patch_embed(x)    # 两层卷积，获得(N, embed_dims[0], H/4, W/4)，两次步长为2做下采样

        x = self.layers[0](x)      #（N,1024,128）     128由下一层输入的特征维度决定，1024是32*32，在上个x(64*64)后经过了下采样
        start_i = 1

        #除了最后一层的每层，都要经过下采样

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        B,_,C=x.size()
        x = x.view(B, 16, 16, C)
        x=x.permute(0, 3, 1, 2)
        x1=self.neck(x)       #(B,256,16,16)
        return x1

    def forward(self, x,mask_ratio):
        if(mask_ratio!=0):
            x=self.random_masking(x,mask_ratio)
        image_imbedding = self.forward_features(x)

        # x=self.avg_pool(image_imbedding).squeeze()
        # x = self.norm_head(x)
        # x = self.head(x)
        return image_imbedding

    def random_masking(self, x, mask_ratio):
        B, _, H,W = x.shape  # batch, length, dim
        x_average=x.mean(dim=(-1,-2),keepdim=True)
        #variance = torch.var(x, dim=(-1, -2))
        #x_average=torch.ones(B, 3, 1, 1).to(x.device)

        mask = torch.bernoulli(torch.full((B, 1, H, W), mask_ratio)).repeat(1, 3, 1,1).to(x.device)
        save=torch.ones(B, 3, H, W).to(x.device)-mask

        #num1_mask=torch.sum(torch.eq(mask.cpu(), 1))
        #num1_save=torch.sum(torch.eq(save.cpu(), 1))
        x_masked=x_average*mask+save*x

        return x_masked
class DCFormerLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0.,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 local_conv_size=3,
                 activation=nn.GELU,
                 out_dim=None,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            DCFormerBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(
                             drop_path, list) else drop_path,
                         local_conv_size=local_conv_size,
                         activation=activation,
                         )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
class DCFormerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 local_conv_size=3,
                 activation=nn.GELU,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = DCMHAttention(dim, head_dim, num_heads,attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C)
            x = self.attn(x)
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size, C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x
class DCMHAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=4, attn_ratio=1, resolution=(16, 16), ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.h= h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.q_norm = RMSnorm(hid_dim=self.d)
        self.k_norm = RMSnorm(hid_dim=self.d)

        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',torch.LongTensor(idxs).view(N, N),persistent=False)

        self.C = nn.Parameter(torch.ones(num_heads, num_heads))
        self.dyn_w_proj = DynamicWeightProjection(num_heads=self.num_heads, query_input_dim=self.key_dim*self.num_heads,
                                                  dynamic_squeeze_ratio=self.num_heads // 2,
                                                  dynamic_w_hidden_dim=self.num_heads * 4)


    def compose(self,att,Q,K,theta):
        W_q1, W_q2, W_k1,W_k2 = theta[0], theta[1], theta[2], theta[3]
        W_qg, W_kg = theta[4], theta[5]  # D_m * H
        dw1, dw2 = self.dw_proj(Q, W_q1, W_q2)
        h = torch.einsum('BHTS,BTRH->BRTS', att, dw1)
        o_qp = torch.einsum('BRTS,BTRH->BHTS', h, dw2)
        dw1, dw2 = self.dw_proj(K, W_k1, W_k2)
        h = torch.einsum('BHTS,BSRH->BRTS', att, dw1)
        o_kp = torch.einsum('BRTS,BSRH->BHTS', h, dw2)
        o_qg = torch.einsum('BHTS,BTH->BHTS', att, torch.tanh(Q @ W_qg))
        o_kg = torch.einsum('BHTS,BSH->BHTS', att, torch.tanh(K @ W_kg))
        return att + o_qp + o_kp + o_qg + o_kg


    def forward(self, x):  # x (B,L,C)
        B, T, _ = x.shape
        x = self.norm(x)          # Normalization
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, T, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, L, num_heads, d)
        q, k = self.q_norm(q), self.k_norm(k)

        #这里去算compose中使用的W_q1, W_q2, W_k1, W_k2
        N, D, I = self.num_heads, self.d, self.dyn_w_proj.dynamic_hidden_dim
        B, T, E = x.shape

        dw_hidden, dd = (x @ self.dyn_w_proj.dw_m).split([2 * 2 * N * (2 * I), 2 * 2 * N * 1], -1)
        dw_hidden = self.dyn_w_proj.dw_hidden_activation(dw_hidden)
        dw_hidden = dw_hidden.view(dw_hidden.shape[:2] + (4, -1))  # B T (4 K) -> B T 4 K  # reshape
        dw = torch.einsum('B T C K, C K D -> B T C D', dw_hidden, self.dyn_w_proj.qkw_m)  # BT4K,4K(MI)->BT4(MI)
        shape = (B, T, 2 * 2, -1, N)  # if project_logits else (B,T,2,N,-1)  # BT(pre/post)(q/k)IN
        w1, w2 = dw.view(shape).split(I, -2)
        w1 = self.dyn_w_proj.dw1_norm(w1)  # BT22IN

        pre_qw1, pre_kw1, post_qw1, post_kw1 = w1.unbind(2)  # BT(2{*2})IN->[BTIN]*4
        pre_qw2, pre_kw2, post_qw2, post_kw2 = w2.unbind(2)
        qkdd = F.tanh(dd).squeeze(-1).view(shape[:-2] + (N,))  # BT(2{*2})N1->BT(2{*2})N
        pre_qdd, pre_kdd, post_qdd, post_kdd = qkdd.unbind(2)  # BT(2{*2})N->[BTN]*4

        pre_dw_args=(pre_qw1,pre_qw2,pre_kw1,pre_kw2,pre_qdd,pre_kdd)
        post_dw_args=(post_qw1,post_qw2,post_kw1,post_kw2,post_qdd,post_kdd)


        q = q.permute(0, 2, 1, 3)  # (B, num_heads, L, d)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)


        out = _atten_context(q, k, v, pre_dw_args, post_dw_args)   #(B,nums_head,T,self.d)
        out = out.transpose(1, 2).reshape(B, T, -1)   #(B,T, nums_head*self.d = dh)
        out = self.proj(out)


        return out
class DynamicWeightProjection(nn.Module):
    def __init__(self, num_heads=32, num_groups=1, query_input_dim=4096, dynamic_squeeze_ratio=1,
                 dynamic_w_hidden_dim=128,dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.query_input_dim = query_input_dim
        self.dynamic_squeeze_ratio = dynamic_squeeze_ratio
        self.dynamic_w_hidden_dim = dynamic_w_hidden_dim
        self.dw_hidden_activation = nn.GELU()
        self.num_heads_per_group = self.num_heads // self.num_groups
        self.dw_activation = nn.Tanh()
        self.dw1_norm = RMSnormNoscale(dim=-1)
        self.pre_proj = CrossHeadProjection('pre', num_heads=self.num_heads)
        self.post_proj = CrossHeadProjection('post', num_heads=self.num_heads)

        dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.dw1 = nn.parameter.Parameter(torch.zeros(self.query_input_dim, self.num_groups, 4, self.dynamic_w_hidden_dim,dtype=dtype))  # (4096, 1, 4, 128)
        G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group
        I = dynamic_hidden_dim * 2
        self.qkw = nn.parameter.Parameter(torch.zeros([G, 4, K, I, M], dtype=dtype))  # (1, 4, 128, 4, 32)
        self.dd = nn.parameter.Parameter(torch.zeros(self.query_input_dim, self.num_groups, self.num_heads_per_group * 4,dtype=dtype))  # (4096, 1, 128)
        self.merge_weights()

    def merge_weights(self):
        self.dw_m = nn.parameter.Parameter(
            torch.cat([self.dw1.reshape(self.query_input_dim, -1), self.dd.squeeze(1)], dim=-1)).to(
            self.dw1.device)  # E,(4*K + K)  K=2*N*I
        self.qkw_m = nn.parameter.Parameter(
            self.qkw.permute(0, 1, 2, 3, 4).reshape(4, self.dynamic_w_hidden_dim, -1)).to(self.dw1.device)  # (4,K,I*M)

    def forward(self, query_vec, KW = None, gen_cache= True):
        #B代表B，T代表输入序列的长度L，D代表输入序列维度
        #G代表num_groups，C是4，代表需要计算的权重数量，一起算了。K代表经过输入W1后较低的维度
        #M代表每个G包含的头数量
        B, T, nums_head,key_dim = query_vec.shape
        query_vec=query_vec.view(B,T,-1)
        dw_hidden = torch.einsum('BTD,DGCK->BTGCK', query_vec, self.dw1)  # C=4 [pre,post]*[query,key]
        dw_hidden = self.dw_hidden_activation(dw_hidden)  # BTGCK
        w1, w2 = torch.split(torch.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, self.qkw), self.qkw.shape[-2] // 2,dim=-2)  # BTGC(2I)M -> [BTGCIM] * 2
        w1 = self.dw1_norm(w1)  # BTGCIM
        pre_qw1, pre_kw1, post_qw1, post_kw1 = unbind(w1, 4, dim=3)  # BTG4IM->[BTGIM]*4
        pre_qw2, pre_kw2, post_qw2, post_kw2 = unbind(w2, 4, dim=3)
        dd = torch.einsum('BTD,DGM->BTGM', query_vec, self.dd)  # BTG(4M)
        dd = self.dw_activation(dd)
        pre_qdd, pre_kdd, post_qdd, post_kdd = torch.split(dd, dd.shape[-1] // 4, dim=-1)  # BTG(4N)->[BTGN]*4
        pre_dw_args = (pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)
        post_dw_args = (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)
        if gen_cache:  # generate KW cache
            pre_kw = torch.einsum('BSGIM, BSGIN->BSMN', pre_kw1, pre_kw2) + torch.diag_embed(
                pre_kdd.squeeze(2))  # merge kw and kdd
            post_kw = torch.einsum('BSGIM, BSGIN->BSMN', post_kw1, post_kw2) + torch.diag_embed(post_kdd.squeeze(2))
            KW = torch.stack((pre_kw, post_kw), dim=-3)  # BSMN,BSMN->BS2MN
        return pre_dw_args, post_dw_args, KW
class RMSnorm(nn.Module):

    def __init__(self, hid_dim=128, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim
        self.hid_dim = hid_dim
        self.epsilon = epsilon
        self.scale = nn.parameter.Parameter(data=torch.ones(self.hid_dim))

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs = normed_inputs * self.scale
        return normed_inputs
def slice_dw(sw, qw1, qw2, kw1, kw2, qdd, kdd, start, stop, kv_start):
    return (sw,
            qw1[:, start: stop] if qw1 is not None else None,
            qw2[:, start: stop] if qw2 is not None else None,
            kw1[:, kv_start: stop] if kw1 is not None else None,
            kw2[:, kv_start: stop] if kw2 is not None else None,
            qdd[:, start: stop] if qdd is not None else None,
            kdd[:, kv_start: stop] if kdd is not None else None)
class RMSnormNoscale(nn.Module):
    def __init__(self, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        return normed_inputs
class CrossHeadProjection(nn.Module):

    def __init__(self, mode, num_heads=16, num_groups=1, dtype=torch.float16, use_sw=False):
        super().__init__()
        self.mode = mode
        self.use_sw = use_sw
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_heads_per_group = self.num_heads // self.num_groups
        if self.use_sw:
            self.w = nn.parameter.Parameter(
                data=torch.zeros(self.num_groups, self.num_heads_per_group, self.num_heads_per_group, dtype=dtype))
        else:
            self.register_buffer('w', torch.eye(self.num_heads_per_group, dtype=dtype).expand(self.num_groups,
                                                                                              self.num_heads_per_group,
                                                                                              self.num_heads_per_group))

    def forward(self, inputs,dws= None,query_vec=None, key_vec=None,proj_w= None,fast_infer=True):
        if proj_w is not None:
            ret = torch.einsum('BNTS,BSNM->BMTS', inputs, proj_w)
        else:
            assert dws is not None
            qw1, qw2, kw1, kw2, qdd, kdd = dws
            inputs = inputs.unsqueeze(1)  # BNTS->BGNTS
            # apply sw
            ret = torch.einsum('BGMTS,GMN->BGNTS', inputs, self.w) if self.use_sw else inputs
            if fast_infer:
                inputs_label = 'BGMTS'
                hidden_sym = 'I';
                hidden_label = inputs_label.replace('M', 'I')  # BGITS
                # apply qw and kw
                for sym, (w1, w2) in zip(['T', 'S'], [(qw1, qw2), (kw1, kw2)]):
                    dw_label = f'B{sym}G{hidden_sym}M'  # w1: BTGIM, dw_label:BTGIM
                    dynamic_hidden_dim = w1.shape[dw_label.index(hidden_sym)]
                    eqn1 = f'{inputs_label},{dw_label}->{hidden_label}'  # 'BGMTS,BTGMI->BGITS'
                    eqn2 = f'{hidden_label},{dw_label}->{inputs_label}'  # 'BGITS,BTGMI->BGMTS'
                    for i in range(dynamic_hidden_dim):
                        hidden = torch.einsum(eqn1.replace(hidden_sym, ''), inputs,
                                              w1[..., i, :])  # BGMTS,BTG(I)M->BGTS
                        out = torch.einsum(eqn2.replace(hidden_sym, ''), hidden,
                                           w2[..., i, :])  # 'BG(I)TS,BTG(I)M->BGMTS'
                        ret = ret + out
                # apply qdd and kdd
                for sym, dd in zip(['T', 'S'], [qdd, kdd]):
                    dd_label = f'B{sym}GM'
                    dout = torch.einsum(f'{inputs_label},{dd_label}->{inputs_label}', inputs,
                                        dd)  # BGMTS,B(T/S)GM->BGMTS
                    ret = ret + dout
            else:
                # apply qw and kw (BTGIN)
                x_inter = torch.einsum('BGNTS, BTGIN->BGTSI', inputs, qw1)
                qw_out = torch.einsum('BGTSI, BTGIN->BGNTS', x_inter, qw2)
                ret = ret + qw_out
                x_inter = torch.einsum('BGNTS, BSGIN->BGTSI', inputs, kw1)
                kw_out = torch.einsum('BGTSI, BSGIN->BGNTS', x_inter, kw2)
                ret = ret + kw_out

                # apply qdd(BTGN) and kdd(BSGN)
                ret = ret + torch.einsum('BGNTS, BTGN->BGNTS', inputs, qdd)
                ret = ret + torch.einsum('BGNTS, BSGN->BGNTS', inputs, kdd)
            ret = ret.squeeze(1)  # BGNTS->BNTS
        return ret
def unbind(ary, n, dim=0):
    return [torch.squeeze(a, dim=dim) for a in torch.split(ary, ary.shape[dim] // n, dim=dim)]
def _atten_context(query, key, value, pre_proj_dw_args, post_proj_dw_args):
    logits = query @ key.transpose(-2, -1)
    logits = _cross_head_proj(logits, *pre_proj_dw_args)
    probs = logits.softmax(-1)
    probs = _cross_head_proj(probs, *post_proj_dw_args)
    o = probs @ value  # BNTS,BNSD->BNTD
    return o
def _cross_head_proj(inputs, qw1, qw2, kw1, kw2, qdd, kdd, loop_over_dynamic_hd=False):
    out = inputs
    for i in range(2):  # qw1.shape[-2]):
        qhidden = (inputs * qw1[..., i, :].transpose(-2, -1).unsqueeze(-1)).sum(1)  # BNTS,(BTN->BNT->BNT1)->BNTS->BTS
        qout = qhidden.unsqueeze(1) * qw2[..., i, :].transpose(-2, -1).unsqueeze(
            -1)  # (BTS->B1TS),(BTN->BNT->BNT1)->BNTS
        out = out + qout
        khidden = (inputs * kw1[..., i, :].transpose(-2, -1).unsqueeze(-2)).sum(1)  # BNTS,(BSN->BNS->BN1S)->BNTS->BTS
        kout = khidden.unsqueeze(1) * kw2[..., i, :].transpose(-2, -1).unsqueeze(
            -2)  # (BTS->B1TS),(BSN->BNS->BNS1)->BNTS
        out = out + kout
    qdout = inputs * qdd.transpose(-2, -1).unsqueeze(-1);
    out = out + qdout  # BNTS,(BTN->BNT->BNT1)->BNTS
    kdout = inputs * kdd.transpose(-2, -1).unsqueeze(-2);
    out = out + kdout  # BNTS,(BSN->BNS->BN1S)->BNTS
    return out



