import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def make_model(args, parent=False):
    return MODEL(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, wn, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            wn(nn.Conv2d(channel, channel // reduction, 1, padding=1 // 2, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(channel // reduction, channel, 1, padding=1 // 2, bias=True)),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def default_conv(in_channels, out_channels, kernel_size, wn, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride == 1:
        padding = kernel_size // 2
    return wn(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups))


## Enhanced spatial attention block(ESA) Layer
class ESA(nn.Module):
    def __init__(self, n_feats, wn, conv=default_conv):
        super(ESA, self).__init__()
        self.f = n_feats // 4
        self.conv1 = conv(n_feats, self.f, kernel_size=1, wn=wn)
        self.conv_f = conv(self.f, self.f, kernel_size=1, wn=wn)
        self.conv_max = conv(self.f, self.f, kernel_size=3, padding=1, wn=wn)
        self.conv2 = conv(self.f, self.f, kernel_size=3, stride=2, padding=0, wn=wn)
        self.conv3 = conv(self.f, self.f, kernel_size=3, padding=1, wn=wn)
        self.conv3_ = conv(self.f, self.f, kernel_size=3, padding=1, wn=wn)
        self.conv4 = conv(self.f, n_feats, kernel_size=1, wn=wn)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


# lightweight lattice block
class LLBlock(nn.Module):
    def __init__(self, num_fea, wn):
        super(LLBlock, self).__init__()
        self.channel1 = num_fea // 2
        self.channel2 = num_fea - self.channel1
        self.convblock = nn.Sequential(
            wn(nn.Conv2d(self.channel1, self.channel1, 3, 1, 3 // 2)),
            nn.LeakyReLU(0.05),
            wn(nn.Conv2d(self.channel1, self.channel1, 3, 1, 3 // 2)),
            nn.LeakyReLU(0.05),
            wn(nn.Conv2d(self.channel1, self.channel1, 3, 1, 3 // 2)),
        )

        self.A_att_conv = CALayer(self.channel1, wn)
        self.B_att_conv = CALayer(self.channel2, wn)

        self.fuse1 = wn(nn.Conv2d(num_fea, self.channel1, 1, 1, 0))
        self.fuse2 = wn(nn.Conv2d(num_fea, self.channel2, 1, 1, 0))
        self.fuse = wn(nn.Conv2d(num_fea, num_fea, 1, 1, 0))

    def forward(self, x):
        x1, x2 = torch.split(x, [self.channel1, self.channel2], dim=1)

        x1 = self.convblock(x1)

        A = self.A_att_conv(x1)
        P = torch.cat((x2, A * x1), dim=1)

        B = self.B_att_conv(x2)
        Q = torch.cat((x1, B * x2), dim=1)

        c = torch.cat((self.fuse1(P), self.fuse2(Q)), dim=1)
        out = self.fuse(c)
        return out


# attention fuse
class AF(nn.Module):
    def __init__(self, num_fea, wn):
        super(AF, self).__init__()
        self.CA1 = CALayer(num_fea, wn)
        self.CA2 = ESA(num_fea, wn)
        self.fuse = nn.Conv2d(num_fea * 2, num_fea, 1)

    def forward(self, x):
        x1 = self.CA1(x)
        x2 = self.CA2(x)
        return self.fuse(torch.cat((x1, x2), dim=1))


#  （编码阶段）
class CFG(nn.Module):
    def __init__(self, n_feats, wn, norm='bn', conv_bias=False):
        super().__init__()
        # 级联部分
        self.conv_d1 = wn(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=3 // 2, dilation=1, bias=conv_bias))
        self.conv_d2 = wn(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=3 // 2, dilation=1, bias=conv_bias))
        self.conv_d3 = wn(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=3 // 2, dilation=1, bias=conv_bias))
        # 空间注意力  生成像素级的权重
        self.attention = AF(n_feats, wn)
        # 定义 channel-attention layer
        # self.channel_attention = CALayer(n_feats,wn)
        self.act = nn.LeakyReLU(0.2, True)
        # 最后一层卷积
        self.conv = wn(
            nn.Conv2d(n_feats * 3, n_feats, kernel_size=1, stride=1, padding=1 // 2, dilation=1, bias=conv_bias))
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(n_feats)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(n_feats)
        else:
            self.norm = None

    def forward(self, x):
        x1 = self.act(self.conv_d1(x))
        x2 = self.act(self.conv_d2(x1 + x))
        x3 = self.act(self.conv_d3(x2 + x))
        x = self.conv(torch.cat([x1, x2, x3], 1)) + x
        x = self.attention(x)
        return x


class AWMS(nn.Module):
    def __init__(
            self, args, scale, n_feats, wn):
        super(AWMS, self).__init__()
        out_feats = scale * scale * args.n_colors
        self.tail_k3 = wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2, dilation=1))
        self.pixelshuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.tail_k3(x)
        x = self.pixelshuffle(x)
        return x


class MODEL(nn.Module):
    def __init__(self, args):
        super(MODEL, self).__init__()
        # hyper-params
        self.args = args
        scale = args.scale[0]
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv = wn(nn.Conv2d(n_feats, n_feats, 3, padding=3 // 2))
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])

        # define head module
        # head = HEAD(args, n_feats, kernel_size, wn)
        head = []
        head.append(
            wn(nn.Conv2d(args.n_colors, n_feats, 3, padding=3 // 2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                CFG(n_feats, wn))
        self.fusion = nn.Sequential(wn(nn.Conv2d(n_feats * n_resblocks, n_feats, 1, padding=1 // 2)),
                                    LLBlock(n_feats, wn), AF(n_feats, wn))
        # define tail module
        out_feats = scale * scale * args.n_colors
        tail = AWMS(args, scale, n_feats, wn)

        skip = []
        skip.append(
            wn(nn.Conv2d(args.n_colors, out_feats, 3, padding=3 // 2))
        )
        skip.append(nn.PixelShuffle(scale))

        self.irb1 = nn.Sequential(wn(nn.Conv2d(3, out_feats, 3, padding=3 // 2)), nn.ReLU(True))
        # self.irb = nn.Sequential(wn(nn.Conv2d(out_feats, out_feats, 3, padding=3 // 2)), nn.ReLU(True))
        self.end = wn(nn.Conv2d(out_feats, 3, 3, padding=3 // 2))
        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = tail
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = (x - self.rgb_mean.cuda() * 255) / 127.5
        s = self.skip(x)
        x = self.head(x)
        res, outputs = x, []
        for resblock in self.body:
            res = resblock(res)
            outputs.append(res)
        f = self.fusion(torch.cat(outputs, 1)) + x
        x = self.tail(f)
        x += s
        #  print("x shape {}".format(x.shape))
        x = self.irb1(x)
        # x = self.irb(x)
        #  x = self.irb(x)
        # x = self.irb(x)
        x = self.end(x)
        x = x * 127.5 + self.rgb_mean.cuda() * 255
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 or name.find('skip') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
