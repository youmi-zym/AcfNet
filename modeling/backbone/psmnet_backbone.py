import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.basic_layers import conv_bn, conv_bn_relu, BasicBlock


class PsmBb(nn.Module):
    """
    Backbone proposed in PSMNet.
    Args:
        in_planes, (int): the channels of input
        batchNorm, (bool): whether use batch normalization layer, default True
    Inputs:
        l_img, (torch.Tensor): left image
        r_img, (torch.Tensor): right image
    Outputs:
        l_fms, (torch.Tensor): left image feature maps
        right, (torch.Tensor): right image feature maps
    """
    def __init__(self, in_planes=3, batchNorm=True):
        super(PsmBb, self).__init__()
        self.in_planes = in_planes
        self.batchNorm = batchNorm

        self.firstconv = nn.Sequential(
            conv_bn_relu(self.batchNorm, self.in_planes,  32, 3, 2, 1, 1, bias=False),
            conv_bn_relu(self.batchNorm, 32, 32, 3, 1, 1, 1, bias=False),
            conv_bn_relu(self.batchNorm, 32, 32, 3, 1, 1, 1, bias=False),
        )

        # For building Basic Block
        self.in_planes = 32

        self.layer1 = self._make_layer(self.batchNorm, BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(self.batchNorm, BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(self.batchNorm, BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(self.batchNorm, BasicBlock, 128, 3, 1, 2, 2)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            conv_bn_relu(self.batchNorm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            conv_bn_relu(self.batchNorm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            conv_bn_relu(self.batchNorm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            conv_bn_relu(self.batchNorm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.lastconv = nn.Sequential(
            conv_bn_relu(self.batchNorm, 320, 128, 3, 1, 1, 1, bias=False),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, dilation=1, bias=False)
        )

    def _make_layer(self, batchNorm, block, out_planes, blocks, stride, padding, dilation):
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = conv_bn(
                batchNorm, self.in_planes, out_planes * block.expansion,
                kernel_size=1, stride=stride, padding=0, dilation=1
            )

        layers = []
        layers.append(
            block(batchNorm, self.in_planes, out_planes, stride, downsample, padding, dilation)
        )
        self.in_planes = out_planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(batchNorm, self.in_planes, out_planes, 1, None, padding, dilation)
            )

        return nn.Sequential(*layers)

    def backbone(self, x):
        output_2_0 = self.firstconv(x)
        output_2_1 = self.layer1(output_2_0)
        output_4_0 = self.layer2(output_2_1)
        output_4_1 = self.layer3(output_4_0)
        output_8 = self.layer4(output_4_1)

        output_branch1 = self.branch1(output_8)
        output_branch1 = F.interpolate(
            output_branch1, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch2 = self.branch2(output_8)
        output_branch2 = F.interpolate(
            output_branch2, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch3 = self.branch3(output_8)
        output_branch3 = F.interpolate(
            output_branch3, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch4 = self.branch4(output_8)
        output_branch4 = F.interpolate(
            output_branch4, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_feature = torch.cat(
            (output_4_0, output_8, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature

    def forward(self, *input):
        if len(input) != 2:
            raise ValueError(
                'expected input length 2 (got {} length input)'.format(len(input))
            )
        l_img, r_img = input

        l_fms = self.backbone(l_img)
        r_fms = self.backbone(r_img)

        return l_fms, r_fms
