import os
from operator import index

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from mamba_ssm import Mamba2
from torchvision.utils import save_image
import torchvision.transforms as transforms
import math
from torch.nn import init





class ECABlock(nn.Module):
    def __init__(self, gamma=2, b=1):
        """
        Efficient Channel Attention (ECA) Module
        Parameters:
        - gamma: scaling factor for kernel size calculation
        - b: additional constant for kernel size calculation
        """
        super(ECABlock, self).__init__()
        self.gamma = gamma
        self.b = b
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = None  # Will initialize in forward()

    def forward(self, x):
        N, C, H, W = x.size()

        # Dynamically calculate kernel size k
        t = int(abs((math.log(C, 2) + self.b) / self.gamma))
        k = t if t % 2 != 0 else t + 1

        # Initialize the Conv1d layer based on the computed kernel size
        if self.conv is None or self.conv.kernel_size[0] != k:
            self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False).to(x.device)

        # Perform ECA operations
        y = self.avg_pool(x)  # Global Average Pooling, shape: [N, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # Conv1d over channels
        y = y.transpose(-1, -2).unsqueeze(-1)  # Reshape back to [N, C, 1, 1]

        return x * y.expand_as(x)

    def _init_weights(self):
        """
        Initialize weights for the Conv1d layer if needed.
        """
        if self.conv is not None:
            init.kaiming_uniform_(self.conv.weight, a=0, mode='fan_in', nonlinearity='relu')



class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Rearrange the tensor for LayerNorm (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Rearrange back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)

class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        self._init_weights()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.pool(x).reshape(batch_size, num_channels)
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.reshape(batch_size, num_channels, 1, 1)
        return x * y
    
    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se_attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        self._init_weights()

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.reshape(batch_size, height * width, -1)

        query = self.split_heads(self.query_dense(x), batch_size)
        key = self.split_heads(self.key_dense(x), batch_size)
        value = self.split_heads(self.value_dense(x), batch_size)
        
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attention_weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embed_size)
        
        output = self.combine_heads(attention)
        
        return output.reshape(batch_size, height, width, self.embed_size).permute(0, 3, 1, 2)

    def _init_weights(self):
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias, 0)
        init.constant_(self.key_dense.bias, 0)
        init.constant_(self.value_dense.bias, 0)
        init.constant_(self.combine_heads.bias, 0)

class Denoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.output_layer = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=1)
        self.res_layer = nn.Conv2d(num_filters, 1, kernel_size=kernel_size, padding=1)
        self.activation = getattr(F, activation)
        self._init_weights()
        self.model = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=1024,  # Model dimension d_model
            d_state=64,  # SSM state expansion factor, typically 64 or 128
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        ).to("cuda")

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))
        print(x4.size())

        x = self.bottleneck(x4)
        # x4 = x4.view(1, 16, 1024)
        # x = self.model(x4)
        # x = x.view(1, 16, 32, 32)
        x = self.up4(x)
        x = self.up3(x + x3)
        x = self.up2(x + x2)
        x = x + x1
        x = self.res_layer(x)
        return torch.tanh(self.output_layer(x + x))
    
    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)
                
class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, bias=True):
        """
        自定义3x3卷积层
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数（即filter个数）
        :param stride: 步长（默认1）
        :param padding: 填充（默认1以保持尺寸不变）
        :param bias: 是否使用偏置
        """
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)

class LYT(nn.Module):
    def __init__(self, filters=32,y_dir="D:\wangbing\LYT-Net\官网Pytorch\output\Lolv2-real\y", cb_dir="D:\wangbing\LYT-Net\官网Pytorch\output\Lolv2-real\cb", cr_dir="D:\wangbing\LYT-Net\官网Pytorch\output\Lolv2-real\cr"):
        super(LYT, self).__init__()
        self.process_y = self._create_processing_layers(filters)
        self.process_cb = self._create_processing_layers(filters)
        self.process_cr = self._create_processing_layers(filters)

        self.y_dir = y_dir
        self.cb_dir = cb_dir
        self.cr_dir = cr_dir

        # 创建文件夹
        os.makedirs(self.y_dir, exist_ok=True)
        os.makedirs(self.cb_dir, exist_ok=True)
        os.makedirs(self.cr_dir, exist_ok=True)

        self.denoiser_cb = Denoiser(filters // 2)
        self.denoiser_cr = Denoiser(filters // 2)
        self.lum_pool = nn.MaxPool2d(8)
        self.conv=Conv3x3(in_channels=1, out_channels=1)
        self.lum_conv1 = Conv3x3(in_channels=filters, out_channels=filters)
        self.lum_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=4)
        self.lum_up = nn.Upsample(scale_factor=8, mode='nearest')
        self.lum_conv = nn.Conv2d(filters, filters, kernel_size=1, padding=0)
        self.ref_conv = nn.Conv2d(filters * 2, filters, kernel_size=1, padding=0)
        self.msef = MSEFBlock(filters)
        self.recombine = nn.Conv2d(filters *2, filters, kernel_size=3, padding=1)
        self.final_adjustments = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self._init_weights()
        self.eca_block = ECABlock(gamma=2, b=1)
        self.lum_model = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=1024,  # Model dimension d_model
            d_state=64,  # SSM state expansion factor, typically 64 or 128
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        ).to("cuda")

    def _create_processing_layers(self, filters):
        return nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _rgb_to_ycbcr(self, image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        
        yuv = torch.stack((y, u, v), dim=1)
        return yuv



    def forward(self, inputs,idx=None):
        if idx is not None:
            print(f"当前批次索引: {idx}")
        print(f"输入图片大小: {inputs.shape}")
        # 如果图片大小是 [1, 3, 400, 600]，则调整大小为 [1, 3, 256, 256]
        if inputs.shape == torch.Size([1, 3, 400, 600]):
            inputs_resized = F.interpolate(inputs, size=(256, 256), mode='bilinear', align_corners=False)
            print(f"调整后的图片大小: {inputs_resized.shape}")
        elif inputs.shape == torch.Size([1, 3, 384, 384]):
            inputs_resized = F.interpolate(inputs, size=(256, 256), mode='bilinear', align_corners=False)
            print(f"调整后的图片大小: {inputs_resized.shape}")
        else:
            # 如果不是 [1, 3, 400, 600]，保持原来的大小
            inputs_resized = inputs
            print(f"图片大小保持不变: {inputs_resized.shape}")
        ycbcr = self._rgb_to_ycbcr(inputs_resized)
        y, cb, cr = torch.split(ycbcr, 1, dim=1)
        #3.0
        y_filename = os.path.join(self.y_dir, f"image_{idx}.png")
        cb_filename = os.path.join(self.cb_dir, f"image_{idx}.png")
        cr_filename = os.path.join(self.cr_dir, f"image_{idx}.png")
        save_image(y, y_filename)
        save_image(cb, cb_filename)
        save_image(cr, cr_filename)
        #3.0

        cb = self.denoiser_cb(cb) + cb
        cr = self.denoiser_cr(cr) + cr
        #cb = self.conv(cb) + cb
        #cr = self.conv(cr) + cr

        y_processed = self.process_y(y)
        cb_processed = self.process_cb(cb)
        cr_processed = self.process_cr(cr)

        ref = torch.cat([cb_processed, cr_processed], dim=1)
        lum = y_processed
        lum_1 = self.lum_pool(lum)
        print(
            f"Before calling lum_model: lum_1 type = {type(lum_1)}, shape = {getattr(lum_1, 'shape', 'Not a tensor')}")
        # lum_1=lum_1.view(1,32,1024)
        # lum_1 = self.lum_model(lum_1)
        # lum_1 =lum_1.view(1,32,32,32)


        lum_1 = self.lum_mhsa(lum_1)
        #lum_1 = self.lum_conv1(lum_1)
        lum_1 = self.lum_up(lum_1)
        lum = lum + lum_1

        ref = self.ref_conv(ref)
        shortcut = ref
        ref = ref + 0.2 * self.lum_conv(lum)
        ref = self.msef(ref)
        ref = ref + shortcut

        recombined = self.recombine(torch.cat([ref, lum], dim=1))
        output = self.final_adjustments(recombined)
        return torch.sigmoid(output)
    
    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)