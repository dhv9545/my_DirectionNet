# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

def equirectangular_padding(x, padding):
    B, C, H, W = x.shape
    top, bottom = padding[0]
    left, right = padding[1]
    semicircle = W // 2
    if top > 0:
        top_padding = torch.flip(torch.roll(x[:, :, :top, :], shifts=semicircle, dims=3), dims=[2])
    else:
        top_padding = torch.empty((B, C, 0, W), device=x.device, dtype=x.dtype)
    if bottom > 0:
        bottom_padding = torch.roll(torch.flip(x[:, :, -bottom:, :], dims=[2]), shifts=semicircle, dims=3)
    else:
        bottom_padding = torch.empty((B, C, 0, W), device=x.device, dtype=x.dtype)
    padded_h = torch.cat([top_padding, x, bottom_padding], dim=2)
    if left > 0:
        left_padding = padded_h[:, :, :, -left:]
    else:
        left_padding = torch.empty((B, C, padded_h.shape[2], 0), device=x.device, dtype=x.dtype)
    if right > 0:
        right_padding = padded_h[:, :, :, :right]
    else:
        right_padding = torch.empty((B, C, padded_h.shape[2], 0), device=x.device, dtype=x.dtype)
    return torch.cat([left_padding, padded_h, right_padding], dim=3)

class BottleneckResidualUnit(nn.Module):
    expansion = 2
    def __init__(self, in_channels, out_filters, strides=1, downsample=None):
        super(BottleneckResidualUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_filters, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters * self.expansion, kernel_size=1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.3)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        y = self.bn1(x)
        y = self.leaky_relu(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.leaky_relu(y)
        y = self.conv2(y)
        y = self.bn3(y)
        y = self.leaky_relu(y)
        y = self.conv3(y)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return y + residual


class SiameseEncoder(nn.Module):
    def __init__(self):
        super(SiameseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer1 = self._make_resblock(64, 2, 128, stride=2)
        self.layer2 = self._make_resblock(128*2, 2, 128, stride=2)
        self.layer3 = self._make_resblock(128*2, 2, 256, stride=2)
        
        main_inplanes = 256 * 2 * 2
        self.main_layer1 = self._make_resblock(main_inplanes, 2, 256, stride=2)
        self.main_layer2 = self._make_resblock(256*2, 2, 256, stride=2)
        
        self.bn = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def _make_resblock(self, inplanes, n_blocks, n_filters, stride=1):
        layers = []
        downsample = None
        if stride != 1 or inplanes != n_filters * BottleneckResidualUnit.expansion:
            downsample = nn.Conv2d(inplanes, n_filters * BottleneckResidualUnit.expansion,
                                   kernel_size=1, stride=stride, bias=False)
            
        layers.append(BottleneckResidualUnit(inplanes, n_filters, stride, downsample))
        inplanes = n_filters * BottleneckResidualUnit.expansion
        for _ in range(1, n_blocks):
            layers.append(BottleneckResidualUnit(inplanes, n_filters, 1))
        return nn.Sequential(*layers)

    def forward(self, img1, img2):
        y1 = self.conv1(img1)
        y1 = self.layer1(y1)
        y1 = self.layer2(y1)
        y1 = self.layer3(y1)
        
        y2 = self.conv1(img2)
        y2 = self.layer1(y2)
        y2 = self.layer2(y2)
        y2 = self.layer3(y2)
        
        y = torch.cat([y1, y2], dim=1)
        y = self.main_layer1(y)
        y = self.main_layer2(y)
        
        y = self.bn(y)
        y = self.leaky_relu(y)
        y = self.global_pool(y)
        return y


class DirectionNet(nn.Module):
    def __init__(self, n_out):
        super(DirectionNet, self).__init__()
        self.encoder = SiameseEncoder()
        
        self.dec1_conv = nn.Conv2d(512, 256, kernel_size=3, padding=0, bias=False)
        # Note: original TF code logically passed 256 to resblock, but since it didn't update tracking var `self.inplanes`
        # which it used to statically dictate downsample (using 512 for inplanes vs 128*2=256), a 1x1 conv was mapped 256 -> 256. 
        # We will faithfully reproduce the output channel numbers.
        self.dec1_res = self._make_resblock(256, 2, 128, force_downsample=True)
        self.dec1_bn = nn.BatchNorm2d(256)
        
        self.dec2_conv = nn.Conv2d(256, 128, kernel_size=3, padding=0, bias=False)
        self.dec2_res = self._make_resblock(128, 2, 64, force_downsample=True)
        self.dec2_bn = nn.BatchNorm2d(128)
        
        self.dec3_conv = nn.Conv2d(128, 64, kernel_size=3, padding=0, bias=False)
        self.dec3_res = self._make_resblock(64, 2, 32, force_downsample=True)
        self.dec3_bn = nn.BatchNorm2d(64)
        
        self.dec4_conv = nn.Conv2d(64, 32, kernel_size=3, padding=0, bias=False)
        self.dec4_res = self._make_resblock(32, 2, 16, force_downsample=True)
        self.dec4_bn = nn.BatchNorm2d(32)
        
        self.dec5_conv = nn.Conv2d(32, 16, kernel_size=3, padding=0, bias=False)
        self.dec5_res = self._make_resblock(16, 2, 8, force_downsample=True)
        self.dec5_bn = nn.BatchNorm2d(16)
        
        self.dec6_conv = nn.Conv2d(16, 8, kernel_size=3, padding=0, bias=False)
        self.dec6_res = self._make_resblock(8, 2, 4, force_downsample=True)
        self.dec6_bn = nn.BatchNorm2d(8)
        
        self.down_channel = nn.Conv2d(8, n_out, kernel_size=1)
        
    def _make_resblock(self, inplanes, n_blocks, n_filters, stride=1, force_downsample=False):
        layers = []
        downsample = None
        if force_downsample or stride != 1 or inplanes != n_filters * BottleneckResidualUnit.expansion:
            downsample = nn.Conv2d(inplanes, n_filters * BottleneckResidualUnit.expansion,
                                   kernel_size=1, stride=stride, bias=False)
            
        layers.append(BottleneckResidualUnit(inplanes, n_filters, stride, downsample))
        inplanes = n_filters * BottleneckResidualUnit.expansion
        for _ in range(1, n_blocks):
            layers.append(BottleneckResidualUnit(inplanes, n_filters, 1))
        return nn.Sequential(*layers)
        
    def _spherical_upsampling(self, x):
        padded = equirectangular_padding(x, ((1, 1), (1, 1)))
        return F.interpolate(padded, scale_factor=2, mode='bilinear', align_corners=False)

    def _decoder_step(self, x, conv, res, bn):
        x = self._spherical_upsampling(x)
        x = conv(x)
        x = res(x)
        x = bn(x)
        x = F.leaky_relu(x, 0.3)
        return x[:, :, 1:-1, 1:-1]

    def forward(self, img1, img2):
        y = self.encoder(img1, img2)
        
        y = self._decoder_step(y, self.dec1_conv, self.dec1_res, self.dec1_bn)
        y = self._decoder_step(y, self.dec2_conv, self.dec2_res, self.dec2_bn)
        y = self._decoder_step(y, self.dec3_conv, self.dec3_res, self.dec3_bn)
        y = self._decoder_step(y, self.dec4_conv, self.dec4_res, self.dec4_bn)
        y = self._decoder_step(y, self.dec5_conv, self.dec5_res, self.dec5_bn)
        y = self._decoder_step(y, self.dec6_conv, self.dec6_res, self.dec6_bn)
        
        return self.down_channel(y)
