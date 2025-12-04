import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import math

class QuasiQuantumDynamicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, groups=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.groups = min(groups, in_ch, out_ch)
        while True:
            if in_ch % self.groups == 0 and out_ch % self.groups == 0:
                break
            self.groups -= 1
            if self.groups == 0:
                raise ValueError(f"Invalid group configuration: in_ch={in_ch}, out_ch={out_ch}")

        self.in_g = in_ch // self.groups
        self.out_g = out_ch // self.groups
        self.w_params = self.out_g * self.in_g * kernel_size**2

        self.qubit_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, 2 * in_ch, 1),
            nn.GELU(),
            nn.Conv2d(2 * in_ch, self.groups * (self.w_params + 1), 1)
        )

        self.base_weights = nn.Parameter(
            torch.randn(self.groups, self.out_g, self.in_g, kernel_size, kernel_size)
        )

        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.groups

        params = self.qubit_generator(x)
        params = params.view(B, g, self.w_params + 1)
        weights, channel_scale = params[..., :-1], params[..., -1]

        weights = weights.view(B, g, self.out_g, self.in_g, self.kernel_size, self.kernel_size)
        dyn_weights = self.base_weights.unsqueeze(0) * weights
        dyn_weights = F.layer_norm(dyn_weights, dyn_weights.shape[-3:])
        dyn_weights = 0.1 * torch.tanh(dyn_weights)
        dyn_weights = dyn_weights.view(B * g, self.out_g, self.in_g, self.kernel_size, self.kernel_size)

        x = x.view(1, B * C, H, W)
        out = F.conv2d(
            input=x,
            weight=dyn_weights.view(-1, self.in_g, self.kernel_size, self.kernel_size),
            padding=self.kernel_size // 2,
            groups=B * g
        )
        out = out.view(B, -1, H, W)
        out = self.bn(out)
        out = out + self.residual_conv(x.view(B, C, H, W))
        return out

class QuantuminspiredTensorEntanglementAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(1) * (self.head_dim ** -0.5))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.entangle = nn.Sequential(
            nn.Conv2d(dim, max(dim // 4, 4), 1),
            nn.GELU(),
            nn.Conv2d(max(dim // 4, 4), num_heads, 1)
        )
        self.proj = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = min(self.window_size, H, W)
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x_padded = F.pad(x, (0, pad_r, 0, pad_b))
        _, _, Hp, Wp = x_padded.shape

        x_window = rearrange(x_padded, 'b c (h hs) (w ws) -> (b h w) c hs ws', hs=ws, ws=ws)
        N = x_window.size(0)

        qkv = self.qkv(x_window).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) hs ws -> b h (hs ws) d', h=self.num_heads), qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        ent = torch.sigmoid(self.entangle(x_window))
        ent = rearrange(ent, 'b h hs ws -> b h (hs ws) 1')
        attn = (attn * ent).softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(N, C, ws, ws)
        out = rearrange(out, '(b h w) c hs ws -> b c (h hs) (w ws)', h=Hp // ws, w=Wp // ws, hs=ws, ws=ws)
        out = out[:, :, :H, :W]
        out = self.bn(self.proj(out))
        return out + x

class AdaptiveMorphology(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid_channels = max(channels // 4, 4)
        self.shared_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.struct_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1),
            nn.GELU(),
            nn.Conv2d(mid_channels, 9 * channels, 1),
            nn.Softmax(dim=1)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.GELU()
        )
        self.residual_conv = nn.Conv2d(channels, channels, 1) if channels != channels else nn.Identity()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        struct = self.struct_gen(x)
        struct = struct.view(B * C, 9, 1, 1).mean(dim=1, keepdim=True)
        struct = F.interpolate(struct, size=(3, 3), mode="bilinear", align_corners=False)

        x_grouped = x.view(1, B * C, H, W)
        dilated = F.conv2d(x_grouped, struct, groups=B * C, padding=1).view(B, C, H, W)
        eroded = F.relu(x - F.relu(self.shared_conv(x)))
        out = self.fusion(torch.cat([dilated, eroded], dim=1))
        out = self.bn(out)
        return out + self.residual_conv(x)



class QRConnection(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # 让 phase_shift 维度与 out_ch 一致
        self.phase_shift = nn.Parameter(torch.randn(out_ch))  

    def forward(self, x_low, x_high):
        B, C, H, W = x_low.shape  # 获取输入的真实通道数
        
        # **动态调整 `freq_transform` 以匹配 `x_high.shape[1]`**
        freq_transform = nn.Sequential(
            nn.Conv2d(C, max(C // 2, 4), 3, padding=1).to(x_low.device),  # 输入通道数 = C
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(max(C // 2, 4), x_high.shape[1], 1).to(x_low.device),  # 输出通道数 = x_high.shape[1]
            nn.Sigmoid()
        )

        # 计算频率掩码
        freq_mask = freq_transform(x_low)  # [B, x_high.shape[1], 1, 1]，确保和 x_high 通道数匹配

        # **确保 phase_shift 维度匹配 x_high**
        if self.phase_shift.shape[0] != x_high.shape[1]: 
            self.phase_shift = nn.Parameter(torch.randn(x_high.shape[1]).to(x_high.device))

        # 频率变换
        x_high = x_high * torch.exp(1j * self.phase_shift[None, :, None, None])

        # **确保 `freq_mask` 和 `x_high` 维度匹配**
        if freq_mask.shape[1] != x_high.shape[1]:
            freq_mask = F.interpolate(freq_mask, size=(x_high.shape[2], x_high.shape[3]), mode="nearest")

        x_high = x_high.real * freq_mask + x_high.imag * (1 - freq_mask)

        return x_low + F.interpolate(x_high, size=x_low.shape[2:], mode='bilinear')





class MQAUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, dims=[8,16,24,32]):
        super().__init__()
        self.in_channels = in_channels  
        self.num_classes = num_classes
        self.dims=dims         
        
        # 初始化卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(dims[0])  # 添加 Batch Normalization              
        )

        self.encoder = nn.ModuleList([
            nn.Sequential(
                QuasiQuantumDynamicConv(dims[0], dims[0], groups=2),
                nn.BatchNorm2d(dims[0]),  # 添加 Batch Normalization
                QuantuminspiredTensorEntanglementAttention(dims[0], num_heads=2),
                AdaptiveMorphology(dims[0])
            ),
            nn.Sequential(
                QuasiQuantumDynamicConv(dims[0], dims[1], groups=2),
                nn.BatchNorm2d(dims[1]),  # 添加 Batch Normalization
                QuantuminspiredTensorEntanglementAttention(dims[1], num_heads=2),
                AdaptiveMorphology(dims[1])
            ),
            nn.Sequential(
                QuasiQuantumDynamicConv(dims[1], dims[2], groups=4),
                nn.BatchNorm2d(dims[2]),  # 添加 Batch Normalization
                QuantuminspiredTensorEntanglementAttention(dims[2], num_heads=4),
                AdaptiveMorphology(dims[2])
            ),
            nn.Sequential(
                QuasiQuantumDynamicConv(dims[2], dims[3], groups=4),
                nn.BatchNorm2d(dims[3]),  # 添加 Batch Normalization
                QuantuminspiredTensorEntanglementAttention(dims[3], num_heads=4),
                AdaptiveMorphology(dims[3])
            )
        ])
        
        # 解码器
        self.decoder = nn.ModuleList([
            QRConnection(dims[3], dims[2]),  # in_ch=dims[3], out_ch=dims[2]
            QRConnection(dims[2], dims[1]),  # in_ch=dims[2], out_ch=dims[1]
            QRConnection(dims[1], dims[0]),  # in_ch=dims[1], out_ch=dims[0]
            QRConnection(dims[0], dims[0])  # in_ch=dims[1], out_ch=dims[0]
        ])
        
        # 特征重组
        self.reorg = nn.ModuleList([ 
            nn.Sequential(
                nn.Conv2d(dims[i] + dims[i + 1], dims[i], 1),
                nn.GELU(),
                nn.BatchNorm2d(dims[i]),  # 添加 Batch Normalization
                nn.Conv2d(dims[i], dims[i], 3, padding=1)
            ) for i in range(3)
        ])
        
        # 输出头
        self.sr_head = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(dims[0]),  # 添加 Batch Normalization
            nn.Conv2d(dims[0], num_classes, 1),
            nn.Sigmoid()  # 输出 Sigmoid 激活函数
        )

    def forward(self, x):
        x = self.init_conv(x)
        enc = []
        
        # 编码过程
        for i, layer in enumerate(self.encoder):
            x = checkpoint(layer, x)
            enc.append(x)
            if i < 3:
                x = F.avg_pool2d(x, 2)
        
        # 解码过程
        for i in range(2, -1, -1):
            x = checkpoint(self.decoder[i], enc[i + 1], x)  # Pass x_high and x_low
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat([x, enc[i]], dim=1)
            x = checkpoint(self.reorg[i], x)
        
        return self.sr_head(x)

if __name__ == "__main__":
    model = MQAUNet(dims=[8,16,24,32])  # 推荐维度配置
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
