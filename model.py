# from FDConv import FDConv
import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
# from contmix import ContMixBlock, LayerNorm2d
import math

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)



class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class FactorizedDeformBranch(nn.Module):
    """
    1x3 deformable -> 3x1 deformable -> 1x1 conv
    用来替换原来的中间 3x3 dilation=2 分支
    """
    def __init__(self, in_channels, out_channels, dilation=2, use_bn=False):
        super(FactorizedDeformBranch, self).__init__()

        self.use_bn = use_bn

        # 1x3 deformable
        self.conv1 = DeformConvPack(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, dilation),
            dilation=(1, dilation),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # 3x1 deformable
        self.conv2 = DeformConvPack(
            out_channels,
            out_channels,
            kernel_size=(3, 1),
            stride=1,
            padding=(dilation, 0),
            dilation=(dilation, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # 1x1 projection
        self.proj = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        self.proj.apply(weights_init_kaiming)
        if use_bn:
            self.bn1.apply(weights_init_kaiming)
            self.bn2.apply(weights_init_kaiming)
            self.bn3.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.proj(x)
        x = self.bn3(x)
        return x

class StandardDeformBranch(nn.Module):
    """
    3x3 standard 2D deformable convolution -> 1x1 projection
    用于与 FactorizedDeformBranch 做公平对比
    """
    def __init__(self, in_channels, out_channels, dilation=2, use_bn=False):
        super(StandardDeformBranch, self).__init__()

        self.use_bn = use_bn

        self.conv = DeformConvPack(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=(dilation, dilation),
            dilation=(dilation, dilation),
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        self.proj = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        self.proj.apply(weights_init_kaiming)

        if use_bn:
            self.bn1.apply(weights_init_kaiming)
            self.bn2.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.proj(x)
        x = self.bn2(x)

        return x

class ReverseFactorizedDeformBranch(nn.Module):
    """
    3x1 deformable -> 1x3 deformable -> 1x1 conv
    Reverse version of FactorizedDeformBranch
    """
    def __init__(self, in_channels, out_channels, dilation=2, use_bn=False):
        super(ReverseFactorizedDeformBranch, self).__init__()

        self.use_bn = use_bn

        # 3x1 deformable
        self.conv1 = DeformConvPack(
            in_channels,
            out_channels,
            kernel_size=(3, 1),
            stride=1,
            padding=(dilation, 0),
            dilation=(dilation, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # 1x3 deformable
        self.conv2 = DeformConvPack(
            out_channels,
            out_channels,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, dilation),
            dilation=(1, dilation),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # 1x1 projection
        self.proj = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        self.proj.apply(weights_init_kaiming)

        if use_bn:
            self.bn1.apply(weights_init_kaiming)
            self.bn2.apply(weights_init_kaiming)
            self.bn3.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.proj(x)
        x = self.bn3(x)

        return x
    
class MSDEE_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MSDEE_module, self).__init__()

        # -------- group 1 --------
        self.FC11 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1,
                              padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)

        # 原来的 3x3 dilation=2，替换成 1x3 -> 3x1 -> 1x1 的可变形方向分支
        self.FC12 = FactorizedDeformBranch(
            in_channels=channel,
            out_channels=channel // 4,
            dilation=2,
            use_bn=True
        )
        # self.FC12 = FactorizedDeformBranch(
        #     in_channels=channel,
        #     out_channels=channel // 4,
        #     dilation=2,
        #     use_bn=True
        # )
        # self.FC12 = ReverseFactorizedDeformBranch(
        #     in_channels=channel,
        #     out_channels=channel // 4,
        #     dilation=2,
        #     use_bn=True
        # )

        self.FC13 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1,
                              padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)

        self.FC1 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)

        # -------- group 2 --------
        self.FC21 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1,
                              padding=1, bias=False, dilation=1)
        self.FC21.apply(weights_init_kaiming)

        self.FC22 = FactorizedDeformBranch(
            in_channels=channel,
            out_channels=channel // 4,
            dilation=2,
            use_bn=True
        )
        # self.FC22 = FactorizedDeformBranch(
        #                 in_channels=channel,
        #                 out_channels=channel // 4,
        #                 dilation=2,
        #                 use_bn=True
        #             )
        # self.FC22 = ReverseFactorizedDeformBranch(
        #     in_channels=channel,
        #     out_channels=channel // 4,
        #     dilation=2,
        #     use_bn=True
        # )

        self.FC23 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1,
                              padding=3, bias=False, dilation=3)
        self.FC23.apply(weights_init_kaiming)

        self.FC2 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC2.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x)) / 3
        x1 = self.FC1(F.relu(x1))

        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x)) / 3
        x2 = self.FC2(F.relu(x2))
        
        out = torch.cat((x, x1, x2), 0)
        out = self.dropout(out)

        return out
    
class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)
        
        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        
        return z

class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(nn.Conv2d(self.low_dim//self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim//self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z

class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)
    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z
 

class CMH_CA(nn.Module):
    """
    Cosine-guided Multi-Head Channel Non-Local block.

    用来替换 DEEN MFA 中的 CNL。

    原始 CNL:
        energy = theta(x_h) @ phi(x_l)

    CMH-CNL:
        1. 将 channel 划分为多个 head；
        2. 每个 head 内独立计算 channel relation；
        3. 引入 cosine relation，减弱 VIS/IR 幅值差异影响；
        4. 输出形式仍然是 W_y + x_h，保持原 CNL residual 结构。

    推荐:
        MFA2: high_dim=512, low_dim=256, num_heads=4
        MFA3: high_dim=1024, low_dim=512, num_heads=8
    """

    def __init__(
        self,
        high_dim,
        low_dim,
        flag=0,
        num_heads=4,
        init_alpha=0.05,
        max_alpha=0.3,
        eps=1e-6
    ):
        super(CMH_CA, self).__init__()

        assert low_dim % num_heads == 0, \
            "low_dim must be divisible by num_heads."

        self.high_dim = high_dim
        self.low_dim = low_dim
        self.flag = flag
        self.num_heads = num_heads
        self.head_dim = low_dim // num_heads
        self.max_alpha = max_alpha
        self.eps = eps

        # 原始 CNL projection，保持不变
        self.g = nn.Conv2d(
            self.low_dim,
            self.low_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.theta = nn.Conv2d(
            self.high_dim,
            self.low_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        if flag == 0:
            self.phi = nn.Conv2d(
                self.low_dim,
                self.low_dim,
                kernel_size=1,
                stride=1,
                padding=0
            )

            self.W = nn.Sequential(
                nn.Conv2d(
                    self.low_dim,
                    self.high_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm2d(high_dim),
            )

        else:
            self.phi = nn.Conv2d(
                self.low_dim,
                self.low_dim,
                kernel_size=1,
                stride=2,
                padding=0
            )

            self.W = nn.Sequential(
                nn.Conv2d(
                    self.low_dim,
                    self.high_dim,
                    kernel_size=1,
                    stride=2,
                    padding=0
                ),
                nn.BatchNorm2d(self.high_dim),
            )

        # cosine relation 的强度，建议 clamp 到 [0, max_alpha]
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        # 每个 head 一个可学习缩放，初始为 1
        self.head_scale = nn.Parameter(torch.ones(num_heads, dtype=torch.float32))

        # 保持原 CNL 的稳定初始化
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        """
        x_h: high-level feature, [B, high_dim, H_h, W_h]
        x_l: low-level feature,  [B, low_dim, H_l, W_l]
        """

        B = x_h.size(0)
        H_l, W_l = x_l.size(2), x_l.size(3)

        # value branch: [B, C, H_l*W_l]
        g_x = self.g(x_l).reshape(B, self.low_dim, -1)

        # query / key branch
        theta_x = self.theta(x_h).reshape(B, self.low_dim, -1)
        phi_x = self.phi(x_l).reshape(B, self.low_dim, -1)

        # reshape to multi-head
        # g:     [B, heads, head_dim, N_l]
        # theta: [B, heads, head_dim, N_h]
        # phi:   [B, heads, head_dim, N_phi]
        g_x = g_x.reshape(B, self.num_heads, self.head_dim, -1)
        theta_x = theta_x.reshape(B, self.num_heads, self.head_dim, -1)
        phi_x = phi_x.reshape(B, self.num_heads, self.head_dim, -1)

        # -------------------------------------------------------
        # 1. grouped raw relation
        # -------------------------------------------------------
        # [B, heads, head_dim, head_dim]
        energy_raw = torch.matmul(
            theta_x,
            phi_x.transpose(-1, -2)
        )

        energy_raw = energy_raw / (energy_raw.size(-1) + self.eps)

        # -------------------------------------------------------
        # 2. cosine-guided relation
        # -------------------------------------------------------
        theta_norm = F.normalize(theta_x, p=2, dim=-1, eps=self.eps)
        phi_norm = F.normalize(phi_x, p=2, dim=-1, eps=self.eps)

        energy_cos = torch.matmul(
            theta_norm,
            phi_norm.transpose(-1, -2)
        )

        alpha = torch.clamp(self.alpha, 0.0, self.max_alpha)

        # 每个 head 的缩放
        head_scale = self.head_scale.view(1, self.num_heads, 1, 1)

        # final relation
        attention = head_scale * (energy_raw + alpha * energy_cos)

        # -------------------------------------------------------
        # 3. apply attention to value
        # -------------------------------------------------------
        # [B, heads, head_dim, N_l]
        y = torch.matmul(attention, g_x)

        # merge heads
        y = y.reshape(B, self.low_dim, -1)

        # 恢复到 low-level feature 的空间尺寸
        y = y.reshape(B, self.low_dim, H_l, W_l)

        W_y = self.W(y)

        z = W_y + x_h

        return z

class RelationPositionGate(nn.Module):
    """
    Relation-guided Position Gate.

    输入:
        energy: [B, Nq, Nk]
        H, W: high-level feature spatial size

    输出:
        gate: [B, 1, H, W]

    作用:
        从 PNL 的 position relation matrix 中生成 query-position gate。
        gate 表示每个 high-level position 的跨层匹配可靠性。
    """

    def __init__(
        self,
        init_beta=0.01,
        max_beta=0.05,
        eps=1e-6
    ):
        super(RelationPositionGate, self).__init__()

        self.max_beta = max_beta
        self.eps = eps

        self.gate = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 1, kernel_size=1, bias=True)
        )

        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # 初始化为 0，使初始 gate = 1
        nn.init.constant_(self.gate[-1].weight, 0.0)
        nn.init.constant_(self.gate[-1].bias, 0.0)

    def forward(self, energy, H, W):
        """
        energy: [B, Nq, Nk]
        """

        B, Nq, Nk = energy.size()

        # 每个 query position 对所有 key positions 的平均相关性
        mean_rel = energy.mean(dim=-1)             # [B, Nq]

        # 每个 query position 最强相关位置的响应
        max_rel = energy.max(dim=-1)[0]            # [B, Nq]

        stat = torch.stack([mean_rel, max_rel], dim=1)  # [B, 2, Nq]

        # 稳定一下尺度，避免 energy 幅值过大
        stat = stat - stat.mean(dim=-1, keepdim=True)
        stat = stat / (stat.std(dim=-1, keepdim=True) + self.eps)

        raw = self.gate(stat)  # [B, 1, Nq]

        beta = torch.clamp(self.beta, 0.0, self.max_beta)

        gate = 1.0 + beta * torch.tanh(raw)
        gate = gate.view(B, 1, H, W)

        return gate

class RPG_SA(nn.Module):
    """
    Relation-guided Position Gated PNL.

    原始 PNL:
        z = W(attention @ g_x) + x_h

    RPG-PNL:
        z = W(attention @ g_x) * relation_position_gate + x_h

    不改变原始 attention，只用 relation gate 轻量调制 residual。
    """

    def __init__(
        self,
        high_dim,
        low_dim,
        reduc_ratio=2,
        init_beta=0.01,
        max_beta=0.05
    ):
        super(RPG_SA, self).__init__()

        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio
        self.inter_dim = low_dim // reduc_ratio

        self.g = nn.Conv2d(
            self.low_dim,
            self.inter_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.theta = nn.Conv2d(
            self.high_dim,
            self.inter_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.phi = nn.Conv2d(
            self.low_dim,
            self.inter_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.W = nn.Sequential(
            nn.Conv2d(
                self.inter_dim,
                self.high_dim,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(high_dim),
        )

        self.rel_gate = RelationPositionGate(
            init_beta=init_beta,
            max_beta=max_beta
        )

        # 保持原 PNL 稳定初始化
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        H, W = x_h.size(2), x_h.size(3)

        # value: [B, Nk, C]
        g_x = self.g(x_l).view(B, self.inter_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta: [B, Nq, C]
        theta_x = self.theta(x_h).view(B, self.inter_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi: [B, C, Nk]
        phi_x = self.phi(x_l).view(B, self.inter_dim, -1)

        energy = torch.matmul(theta_x, phi_x)      # [B, Nq, Nk]
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)           # [B, Nq, C]
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.inter_dim, H, W)

        W_y = self.W(y)

        # 关键：gate 来自 PNL 自己的 relation matrix
        pos_gate = self.rel_gate(energy, H, W)

        W_y = W_y * pos_gate

        z = W_y + x_h

        return z

class DGCLRM_block(nn.Module):
    def __init__(
        self,
        high_dim,
        low_dim,
        flag=0,
        cnl_heads=4,
        cnl_alpha=0.01,
        cnl_max_alpha=0.1,
        pnl_beta=0.01,
        pnl_max_beta=0.05,
        reduc_ratio=2
    ):
        super(DGCLRM_block, self).__init__()

        self.CNL = CMH_CA(
            high_dim=high_dim,
            low_dim=low_dim,
            flag=flag,
            num_heads=cnl_heads,
            init_alpha=cnl_alpha,
            max_alpha=cnl_max_alpha
        )

        self.PNL = RPG_SA(
            high_dim=high_dim,
            low_dim=low_dim,
            reduc_ratio=reduc_ratio,
            init_beta=pnl_beta,
            max_beta=pnl_max_beta
        )

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z

class DGCLRM_block_Option(nn.Module):
    """
    Switchable MFA block for ablation.

    use_cmh=False, use_rpg=False:
        original CNL -> original PNL

    use_cmh=True, use_rpg=False:
        CMH-CNL -> original PNL

    use_cmh=False, use_rpg=True:
        original CNL -> RPG-PNL

    use_cmh=True, use_rpg=True:
        CMH-CNL -> RPG-PNL
    """

    def __init__(
        self,
        high_dim,
        low_dim,
        flag=0,
        cnl_heads=4,
        cnl_alpha=0.01,
        cnl_max_alpha=0.1,
        pnl_beta=0.01,
        pnl_max_beta=0.05,
        reduc_ratio=2,
        use_cmh=True,
        use_rpg=True
    ):
        super(DGCLRM_block_Option, self).__init__()

        if use_cmh:
            self.CNL = CMH_CNL(
                high_dim=high_dim,
                low_dim=low_dim,
                flag=flag,
                num_heads=cnl_heads,
                init_alpha=cnl_alpha,
                max_alpha=cnl_max_alpha
            )
        else:
            self.CNL = CNL(
                high_dim=high_dim,
                low_dim=low_dim,
                flag=flag
            )

        if use_rpg:
            self.PNL = RPG_PNL(
                high_dim=high_dim,
                low_dim=low_dim,
                reduc_ratio=reduc_ratio,
                init_beta=pnl_beta,
                max_beta=pnl_max_beta
            )
        else:
            self.PNL = PNL(
                high_dim=high_dim,
                low_dim=low_dim,
                reduc_ratio=reduc_ratio
            )

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z
        

class RelationChannelGate(nn.Module):
    """
    Relation-guided Channel Gate.

    输入:
        energy: [B, C, C]

    输出:
        gate: [B, C, 1, 1]

    含义:
        从 CNL 的 channel relation matrix 中判断每个输出通道的聚合可靠性。
    """

    def __init__(
        self,
        channels,
        init_beta=0.01,
        max_beta=0.05,
        eps=1e-6
    ):
        super(RelationChannelGate, self).__init__()

        self.channels = channels
        self.max_beta = max_beta
        self.eps = eps

        self.gate = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 1, kernel_size=1, bias=True)
        )

        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # 初始 raw=0, gate=1
        nn.init.constant_(self.gate[-1].weight, 0.0)
        nn.init.constant_(self.gate[-1].bias, 0.0)

    def forward(self, energy):
        """
        energy: [B, C, C]
        """

        B, C, _ = energy.size()

        # 每个 query channel 对所有 key channels 的平均关系
        mean_rel = energy.mean(dim=-1)       # [B, C]

        # 每个 query channel 的最强 channel 匹配
        max_rel = energy.max(dim=-1)[0]      # [B, C]

        stat = torch.stack([mean_rel, max_rel], dim=1)  # [B, 2, C]

        # 稳定尺度
        stat = stat - stat.mean(dim=-1, keepdim=True)
        stat = stat / (stat.std(dim=-1, keepdim=True) + self.eps)

        raw = self.gate(stat)  # [B, 1, C]

        beta = torch.clamp(self.beta, 0.0, self.max_beta)

        gate = 1.0 + beta * torch.tanh(raw)
        gate = gate.view(B, C, 1, 1)

        return gate

class embed_net(nn.Module):
    def __init__(self,  class_num, dataset, arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        
        self.dataset = dataset
        if self.dataset == 'regdb': # For regdb dataset, we remove the MFA3 block and layer4.
            pool_dim = 1024
            self.DEE = MSDEE_module(512)
            self.MFA1 = MFA_block(256, 64, 0)
            self.MFA2 = DGCLRM_block(
                            high_dim=512,
                            low_dim=256,
                            flag=1,
                            cnl_heads=4,
                            cnl_alpha=0.01,
                            cnl_max_alpha=0.3,
                            pnl_beta=0.001,
                            pnl_max_beta=0.013,
                            reduc_ratio=2
                        )
        else:
            pool_dim = 2048
            self.DEE = MSDEE_module(1024)

            self.MFA1 = MFA_block(256, 64, 0)

            self.MFA2 = DGCLRM_block(
                            high_dim=512,
                            low_dim=256,
                            flag=1,
                            cnl_heads=4,
                            cnl_alpha=0.02,
                            cnl_max_alpha=0.3,
                            pnl_beta=0.001,
                            pnl_max_beta=0.013,
                            reduc_ratio=2
                        )
            # self.MFA2 = DGCLRM_block_Option(
            #                 high_dim=512,
            #                 low_dim=256,
            #                 flag=1,
            #                 cnl_heads=4,
            #                 cnl_alpha=0.02,
            #                 cnl_max_alpha=0.3,
            #                 pnl_beta=0.001,
            #                 pnl_max_beta=0.013,
            #                 reduc_ratio=2,
            #                 use_cmh=True,
            #                 use_rpg=True
            #             )
                        
            self.MFA3 = MFA_block(1024, 512, 1)


        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        x_ = x
        x = self.base_resnet.base.layer1(x_)
        x_ = self.MFA1(x, x_)
        x = self.base_resnet.base.layer2(x_)
        x_ = self.MFA2(x, x_)
        if self.dataset == 'regdb':  # For regdb dataset, we remove the MFA3 block and layer4.
            x_ = self.DEE(x_)
            x = self.base_resnet.base.layer3(x_)
        else:
            x = self.base_resnet.base.layer3(x_)
            x_ = self.MFA3(x, x_)
            x_ = self.DEE(x_)
            x = self.base_resnet.base.layer4(x_)
        
        xp = self.avgpool(x)
        x_pool = xp.view(xp.size(0), xp.size(1))
        
        feat = self.bottleneck(x_pool)

        if self.training:
            xps = xp.view(xp.size(0), xp.size(1), xp.size(2)).permute(0, 2, 1)
            xp1, xp2, xp3 = torch.chunk(xps, 3, 0)
            xpss = torch.cat((xp2, xp3), 1)
            loss_ort = torch.triu(torch.bmm(xpss, xpss.permute(0, 2, 1)), diagonal = 1).sum() / (xp.size(0))

            return x_pool, self.classifier(feat), loss_ort
        else:
            return self.l2norm(x_pool), self.l2norm(feat)
