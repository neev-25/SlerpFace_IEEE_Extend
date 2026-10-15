# based on:
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/backbone/model_irse.py
# Derived from Tencent/TFace SlerpFace at 171ec1434aca0972e111663ac646c9c1e3e44b5b.
# Upstream license/attribution: third_party/tface/License.txt and LICENSE.face-evoLVe.txt.
# Locally modified: validation, stable normalization, deep channels, initialization,
# and the explicit pinned-backbone import; see docs/THIRD_PARTY_NOTICES.md.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from third_party.tface.torchkit.backbone.common import initialize_weights, Flatten
from third_party.tface.torchkit.backbone.model_irse import (
    BasicBlockIR, BottleneckIR, BasicBlockIRSE, BottleneckIRSE, get_blocks
)

def normal_init(m, mean, std):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()

class XCosAttention(nn.Module):
    def __init__(self, group_size, use_softmax=True, softmax_t=1, chw2hwc=True):
        super(XCosAttention, self).__init__()
        if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
            raise ValueError('group_size must be a positive integer')
        if not math.isfinite(softmax_t) or softmax_t <= 0:
            raise ValueError('softmax_t must be finite and greater than zero')
        self.embedding_net = nn.Sequential(
            nn.Conv2d(group_size, group_size, 3, padding=1),
            nn.BatchNorm2d(group_size),
            nn.PReLU())
        self.attention = nn.Sequential(
            nn.Conv2d(group_size, group_size, 3, padding=1),
            nn.BatchNorm2d(group_size),
            nn.PReLU(),
            nn.Conv2d(group_size, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.USE_SOFTMAX = use_softmax
        self.SOFTMAX_T = softmax_t
        self.chw2hwc = chw2hwc

    def softmax(self, x, T=1):
        if not math.isfinite(T) or T <= 0:
            raise ValueError('temperature must be finite and greater than zero')
        return F.softmax((x / T).reshape(x.size(0), x.size(1), -1), 2).view_as(x)

    def divByNorm(self, x):
        shifted = x - x.amin(dim=(2, 3), keepdim=True)
        total = shifted.sum(dim=(2, 3), keepdim=True)
        # A constant map contains no preference: use uniform attention, without
        # mutating the caller's tensor or dividing by zero.
        return torch.where(
            total > 0,
            shifted / total.clamp_min(torch.finfo(x.dtype).tiny),
            torch.full_like(x, 1.0 / (x.size(2) * x.size(3))),
        )

    def forward(self, feat_grid):
        proj_embed = self.embedding_net(feat_grid)
        attention_weights = self.attention(proj_embed)
        if self.USE_SOFTMAX:
            attention_weights = self.softmax(attention_weights, self.SOFTMAX_T)
        else:
            attention_weights = self.divByNorm(attention_weights)
        if self.chw2hwc:
            attention_weights = attention_weights.permute(0, 2, 3, 1)
        return attention_weights

    def weight_init(self, mean, std):
        self.apply(lambda module: normal_init(module, mean, std))

class SlerpFace(nn.Module):
    def __init__(self, input_size, num_layers, mode='ir', group_size=16):
        super(SlerpFace, self).__init__()
        if tuple(input_size) not in ((112, 112), (224, 224)):
            raise ValueError('input_size must be [112, 112] or [224, 224]')
        if num_layers not in [18, 34, 50, 100, 152, 200]:
            raise ValueError('num_layers must be 18, 34, 50, 100, 152 or 200')
        if mode not in ['ir', 'ir_se']:
            raise ValueError('mode must be ir or ir_se')
        if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
            raise ValueError('group_size must be a positive integer')
        self.input_size = tuple(input_size)

        # 初始化输入层
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        # get blocks
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            unit_module = BasicBlockIRSE if mode == 'ir_se' else BasicBlockIR
            output_channel = 512
        else:
            unit_module = BottleneckIRSE if mode == 'ir_se' else BottleneckIR
            output_channel = 2048

        # construct body
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
        self.body = nn.Sequential(*modules)

        # output layer
        if input_size[0] == 112:
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(output_channel),
                nn.Dropout(0.4),
                Flatten(),
                nn.Linear(output_channel * 7 * 7, 512),
                nn.BatchNorm1d(512, affine=False)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(output_channel),
                nn.Dropout(0.4),
                Flatten(),
                nn.Linear(output_channel * 14 * 14, 512),
                nn.BatchNorm1d(512, affine=False)
            )

        # Xcos module
        self.group_cov = nn.Sequential(
            nn.Conv2d(output_channel, group_size, (1, 1), 1, 0),
            nn.BatchNorm2d(group_size),
            nn.PReLU(group_size)
        )
        self.group_attention = XCosAttention(group_size=group_size, use_softmax=True, softmax_t=1, chw2hwc=True)

        # initialize weights
        initialize_weights(self.modules())

    def get_body_output(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, *self.input_size):
            raise ValueError(f'input must have shape (N, 3, {self.input_size[0]}, {self.input_size[1]})')
        x = self.input_layer(x)
        return self.body(x)

    def forward(self, x, flip=True):
        x_body = self.get_body_output(x)
        x_body_cov = self.group_cov(x_body)
        x_vector = self.output_layer(x_body)
        
        if flip:
            x_flip = torch.flip(x, dims=[3])
            flip_body = self.get_body_output(x_flip)
            flip_body_cov = self.group_cov(flip_body)
            return x_vector, x_body_cov + flip_body_cov
        return x_vector, x_body_cov

    def gen_weight_map(self, x):
        return self.group_attention(x)

    def gen_vector_feature(self, x):
        return self.output_layer(self.get_body_output(x))

    def gen_group_feature(self, x, flip=True):
        x_body = self.get_body_output(x)
        x_cov = self.group_cov(x_body)
        if flip:
            x_flip = torch.flip(x, dims=[3])
            flip_body = self.get_body_output(x_flip)
            flip_cov = self.group_cov(flip_body)
            return x_cov + flip_cov
        return x_cov

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = SlerpFace(
        input_size=[112, 112],
        num_layers=50,
        group_size=16
    ).to(device).eval()
    x = torch.randn(1, 3, 112, 112, device=device)
    group_feature = backbone.gen_group_feature(x)
    print(group_feature.shape)
