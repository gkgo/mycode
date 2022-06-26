import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.functional import unfold, pad
import warnings
from timm.models.layers import trunc_normal_, DropPath
# from torch import nn

# class SCR(nn.Module):
#     def __init__(self, planes=[640, 64, 64, 64, 640], stride=1, ksize=3, do_padding=False, bias=False):
#         super(SCR, self).__init__()
#         self.ksize = _quadruple(ksize) if isinstance(ksize, int) else ksize  # 4倍 isinstance() 函数来判断一个对象是否是一个已知的类型（是否与int一个类型）
#         padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)
#
#         self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
#                                         nn.BatchNorm2d(planes[1]),
#                                         nn.ReLU(inplace=True))
#         self.conv1 = nn.Sequential(nn.Conv3d(planes[1], planes[2], (1, self.ksize[2], self.ksize[3]),
#                                              stride=stride, bias=bias, padding=padding1),
#                                    nn.BatchNorm3d(planes[2]),
#                                    nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(nn.Conv3d(planes[2], planes[3], (1, self.ksize[2], self.ksize[3]),
#                                              stride=stride, bias=bias, padding=padding1),
#                                    nn.BatchNorm3d(planes[3]),
#                                    nn.ReLU(inplace=True))
#         self.conv1x1_out = nn.Sequential(
#             nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
#             nn.BatchNorm2d(planes[4]))
#
#     def forward(self, x):
#         b, c, h, w, u, v = x.shape
#         x = x.view(b, c, h * w, u * v)
#
#         x = self.conv1x1_in(x)   # [80, 640, hw, 25] -> [80, 64, hw, 25]
#
#         c = x.shape[1]
#         x = x.view(b, c, h * w, u, v)
#         x = self.conv1(x)  # [80, 64, hw, 5, 5] --> [80, 64, hw, 3, 3]
#         x = self.conv2(x)  # [80, 64, hw, 3, 3] --> [80, 64, hw, 1, 1]
#
#         c = x.shape[1]
#         x = x.view(b, c, h, w)
#         x = self.conv1x1_out(x)  # [80, 64, h, w] --> [80, 640, h, w]
#         return x


# class SelfCorrelationComputation(nn.Module):
#     def __init__(self, kernel_size=(5, 5), padding=2):
#         super(SelfCorrelationComputation, self).__init__()
#         self.kernel_size = kernel_size
#         self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         x = self.relu(x)
#         x = F.normalize(x, dim=1, p=2)
#         identity = x
#
#         x = self.unfold(x)  # 提取出滑动的局部区域块，这里滑动窗口大小为5*5，步长为1
#         # b, cuv, h, w  （80,640*5*5,5,5)
#         x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)  # b, c, u, v, h, w
#         x = x * identity.unsqueeze(2).unsqueeze(2)  # 通过unsqueeze增维使identity和x变为同维度  公式（1）
#         # b, c, u, v, h, w * b, c, 1, 1, h, w
#         x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v
#         # torch.contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列
#         return x




class SelfCorrelationComputation(nn.Module):
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = 2 * kernel_size - 1

        self.relu = nn.ReLU(inplace=True)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpb = nn.Parameter(torch.zeros(num_heads, self.rpb_size, self.rpb_size))
        trunc_normal_(self.rpb, std=.02)
        # RPB implementation by @qwopqwop200
        self.idx_h = torch.arange(0, kernel_size)
        self.idx_w = torch.arange(0, kernel_size)
        self.idx_k = ((self.idx_h.unsqueeze(-1) * self.rpb_size) + self.idx_w).view(-1)
        warnings.warn("This is the legacy version of NAT -- it uses unfold+pad to produce NAT, and is highly inefficient.")
        self.bn1 = nn.BatchNorm2d(640)
        # self.conv1x1_in = nn.Sequential(nn.Conv2d(640, 64, kernel_size=1, bias=False, padding=0),
        #                                 nn.BatchNorm2d(640),
        #                                 nn.ReLU(inplace=True))
        # self.conv1x1_out = nn.Sequential(
        #     nn.Conv2d(64, 640, kernel_size=1, bias=False, padding=0),
        #     nn.BatchNorm2d(64))


    def apply_pb(self, attn, height, width):

        num_repeat_h = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_w = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_h[self.kernel_size//2] = height - (self.kernel_size-1)
        num_repeat_w[self.kernel_size//2] = width - (self.kernel_size-1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*self.kernel_size-1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        # Index flip
        # Our RPB indexing in the kernel is in a different order, so we flip these indices to ensure weights match.
        bias_idx = torch.flip(bias_idx.reshape(-1, self.kernel_size**2), [0])
        return attn + self.rpb.flatten(1, 2)[:, bias_idx].reshape(self.num_heads, height * width, 1, self.kernel_size ** 2).transpose(0, 1)

    def forward(self, x):
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        x = x.permute(0, 2, 3, 1).contiguous()
        # x = self.norm1(x)
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(self.kernel_size ** 2)
        pad_l = pad_t = pad_r = pad_b = 0
        Ho, Wo = H, W
        if N <= num_tokens:
            if self.kernel_size > W:
                pad_r = self.kernel_size - W
            if self.kernel_size > H:
                pad_b = self.kernel_size - H
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            B, H, W, C = x.shape
            N = H * W
            assert N == num_tokens, f"Something went wrong. {N} should equal {H} x {W}!"
        x = self.qkv(x).reshape(B, H, W, 3 * C)  # (80,5,5,1920)
        q, x = x[:, :, :, :C], x[:, :, :, C:]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads, 1).transpose(3, 4) * self.scale  # (80,25,1,1,640)
        pd = self.kernel_size - 1
        pdr = pd // 2

        if self.mode == 0:
            x = x.permute(0, 3, 1, 2).flatten(0, 1)
            x = x.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1).permute(0, 3, 4, 1, 2)
            x = pad(x, (pdr, pdr, pdr, pdr, 0, 0), 'replicate')
            x = x.reshape(B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)
        elif self.mode == 1:
            Hr, Wr = H - pd, W - pd
            x = unfold(x.permute(0, 3, 1, 2),
                       kernel_size=(self.kernel_size, self.kernel_size),
                       stride=(1, 1),
                       padding=(0, 0)).reshape(B, 2 * C * num_tokens, Hr, Wr)
            x = pad(x, (pdr, pdr, pdr, pdr), 'replicate').reshape(
                B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)  # (2,80,25,1,9,640)
        else:
            raise NotImplementedError(f'Mode {self.mode} not implemented for NeighborhoodAttention2D.')
        k, v = x[0], x[1]

        attn = (q @ k.transpose(-2, -1))  # B x N x H x 1 x num_tokens
        attn = self.apply_pb(attn, H, W)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # B x N x H x 1 x C # (80,25,1,1,640)
        x = x.reshape(B, H, W, C)  # (80,5,5,640)
        if pad_r or pad_b:
            x = x[:, :Ho, :Wo, :]
        x = self.proj_drop(self.proj(x))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.bn1(x)

        # x = self.conv1x1_in(x)
        # x = self.conv1x1_out(x)
        # x = self.proj_drop(x)

        # return self.proj_drop(self.proj(x))

        return x

