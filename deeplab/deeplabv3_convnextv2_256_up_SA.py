import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.convnextv2_qinglianghua import convnextv2_base
from nets.coordatt_up import CoordAtt
class ConvNeXtBackbone(nn.Module):
    """ConvNeXt Backbone for DeepLab with Pretrained Weights."""

    def __init__(self, downsample_factor=8, pretrained=True):
        super(ConvNeXtBackbone, self).__init__()

        model = convnextv2_base()

        self.model = model
        # 设置特征通道数
        self.low_level_channels = 128 # feat1 通道数
        self.in_channels =256# feat4 通道数

    def forward(self, x):
        # 使用 backbone 获取特征图，返回的是一个列表，包含了四个阶段的特征图
        feats = self.model.forward(x)

        # 解包特征图
        # feat1, feat2, feat3, feat4 = feats

        low_level_features = feats[0]
        x = feats[1]

        return low_level_features, x




class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        # 各分支使用深度可分离卷积减少参数量
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, 1, padding=6 * rate, dilation=6 * rate, groups=dim_in, bias=True),
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, 1, padding=12 * rate, dilation=12 * rate, groups=dim_in, bias=True),
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, 1, padding=18 * rate, dilation=18 * rate, groups=dim_in, bias=True),
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.ca = CoordAtt(dim_out,dim_out)  # 通道注意力
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()

        # 多尺度特征提取
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = self.branch5(x)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # 特征拼接与通道注意力
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        feature_cat = self.conv_cat(feature_cat)
        result= self.ca(feature_cat)
        return result

# class AdaptiveEdgeFusion(nn.Module):
#     """维度修复版边缘融合模块 (兼容TorchScript版本)"""
#
#     def __init__(self, low_ch, high_ch):
#         super().__init__()
#         # 边缘检测路径
#         self.edge_path = nn.Sequential(
#             nn.Conv2d(low_ch, 64, 3, padding=1, groups=4),
#             nn.BatchNorm2d(64),
#             AdaptiveCannyDetector(),
#             nn.Conv2d(64, high_ch, 1)
#         )
#         # 动态权重生成
#         self.att_conv = nn.Sequential(
#             nn.Conv2d(high_ch * 2, high_ch // 4, 3, padding=1),
#             nn.ReLU(),
#             CoordAtt(high_ch // 4,high_ch // 4),
#             nn.Conv2d(high_ch // 4, 2, 1),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, low_feat, high_feat):
#         # 维度校验（处理可能的5维输入）
#         if low_feat.dim() == 5:
#             low_feat = low_feat.squeeze(1)  # 从[B,1,C,H,W]变为[B,C,H,W]
#
#         # 生成边缘特征
#         edge_feat = self.edge_path(low_feat)
#
#         # 动态尺寸对齐 (关键修改点)
#         high_feat = F.interpolate(
#             input=high_feat,
#             size=edge_feat.shape[-2:],  # 直接对齐到边缘特征尺寸
#             mode='bilinear',
#             align_corners=False
#         )
#
#         # 特征融合
#         combined = torch.cat([high_feat, edge_feat], dim=1)
#         att_weights = self.att_conv(combined)
#
#         # 加权融合
#         return high_feat * att_weights[:, 0:1] + edge_feat * att_weights[:, 1:2]
class AdaptiveCannyDetector(nn.Module):
    """可自适应学习Canny边缘检测的模块"""

    def __init__(self, in_channels):  # 添加输入通道参数
        super().__init__()
        self.in_channels = in_channels

        # 高斯模糊层
        self.gauss = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)  # 替换64为in_channels

        # 阈值预测网络
        self.thresh_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 16, 1),  # 输入通道改为in_channels
            nn.ReLU(),
            nn.Conv2d(16, 2, 1),
            nn.Sigmoid()
        )

        # Sobel初始化
        self._init_sobel_weights()

    def _init_sobel_weights(self):
        # X方向梯度
        self.gx = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1,
                            groups=self.in_channels, bias=False)  # 替换64为in_channels
        gx_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.gx.weight.data = gx_kernel.repeat(self.in_channels, 1, 1, 1)  # 替换64为in_channels

        # Y方向梯度
        self.gy = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1,
                            groups=self.in_channels, bias=False)  # 替换64为in_channels
        gy_kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        self.gy.weight.data = gy_kernel.repeat(self.in_channels, 1, 1, 1)  # 替换64为in_channels

    def forward(self, x):
        # 与原代码一致，无需修改
        x = self.gauss(x)
        gx = self.gx(x)  # [B, in_channels, H, W]
        gy = self.gy(x)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        # 动态阈值
        thresh = self.thresh_net(edge)  # [B,2,1,1]
        thresh_low = thresh[:, 0:1]  # [B,1,1,1]
        thresh_high = thresh[:, 1:2]  # [B,1,1,1]

        mask = (edge > thresh_low) & (edge < thresh_high)
        return mask.float() * edge

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="convnextv2", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "convnextv2":
            self.backbone = ConvNeXtBackbone(pretrained=pretrained)
            in_channels = 256
            low_level_channels = 128
        else:
            raise ValueError(f"Unsupported backbone - {backbone}")

        self.aspp = ASPP(dim_in=in_channels, dim_out=128, rate=16 // downsample_factor)

        # 低级特征处理（维度对齐）
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels,96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

        )
        self.AdaptiveCanny=AdaptiveCannyDetector(96)



        # 特征融合（参数优化）
        self.cat_conv = nn.Sequential(
            nn.Conv2d(96+96+ 128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.cls_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        H, W = x.size()[2:]
        low_level, x_high = self.backbone(x)
        x_aspp = self.aspp(x_high)

        # 低级特征处理
        low_level = self.shortcut_conv(low_level)

        # 边缘融合（尺寸对齐）
        x_aspp_up = F.interpolate(x_aspp,  low_level.shape[2:], mode='bilinear', align_corners=True)
        low_level_edge=self.AdaptiveCanny(low_level)

        # 最终融合
        x = torch.cat([low_level,low_level_edge, x_aspp_up], dim=1)
        x = self.cat_conv(x)
        return F.interpolate(self.cls_conv(x), (H, W), mode='bilinear')