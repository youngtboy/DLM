import torch
from src.backbone.pvtv2 import pvt_v2_b2_li
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class Residual_Block(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
    ):
        super(Residual_Block, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x) :
        x = self.conv(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class MSModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(MSModule, self).__init__()
        self.s_conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())
        self.s_conv_2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())
        self.m_conv_1 = nn.Sequential(nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())
        self.m_conv_2 = nn.Sequential(nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())

        self.s_conv_3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())

        self.m_conv_3 = nn.Sequential(nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())

        self.s_residual_block = Residual_Block(out_channels, out_channels)
        self.m_residual_block = Residual_Block(out_channels*4, out_channels)

    def forward(self, img_feature, label_feature1, label_feature2, label_feature3):
        label_feature1 = self.s_conv_1(label_feature1)
        label_feature2 = self.s_conv_2(label_feature2)
        label_feature3 = self.s_conv_3(label_feature3)
        m = torch.cat((self.s_residual_block(img_feature), label_feature1, label_feature2, label_feature3), dim=1)
        m = self.m_residual_block(m)
        s1 = self.m_conv_1(torch.cat((m, label_feature1), dim=1))
        s2 = self.m_conv_2(torch.cat((m, label_feature2), dim=1))
        s3 = self.m_conv_3(torch.cat((m, label_feature3), dim=1))
        return s1, s2, s3


class DLM(nn.Module):
    """
        Depth-inspired Label Mining (DLM) for Unsupervised RGB-D Salient Object Detection
    """
    def __init__(self,):
        super(DLM, self).__init__()
        self.rgb_encoder = pvt_v2_b2_li()
        self.rgb_rfb = RFB_modified(512, 512)
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(512, 320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),)
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),)

        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),)

        self.seg_head = nn.Conv2d(64, 1, kernel_size=1)

        self.ms1 = MSModule(3, 64, stride=2)
        self.ms2 = MSModule(64, 128, stride=2)
        self.ms3 = MSModule(128, 320, stride=2)
        self.ms4 = MSModule(320, 512, stride=1)

        self.uc1 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1), nn.ReLU())
        self.uc2 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1), nn.ReLU())
        self.uc3 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1), nn.ReLU())

    def forward(self, image, label1, label2, label3):
        x_rgb_list = self.rgb_encoder(image)
        x_rgb = self.rgb_rfb(x_rgb_list[-1])
        x = x_rgb
        x = self.decoder_stage1(x+x_rgb_list[-1])
        x = self.decoder_stage2(x+x_rgb_list[-2])
        x = self.decoder_stage3(x+x_rgb_list[-3])
        x = self.decoder_stage4(x+x_rgb_list[-4])
        x = self.seg_head(x)

        if (label1 is not None) and (label2 is not None) and (label3 is not None):
            l_1, l_2, l_3 = self.ms1(F.interpolate(x_rgb_list[0], scale_factor=4, mode="bilinear", align_corners=True),
                           label1, label2, label3)
            l_1, l_2, l_3 = self.ms2(F.interpolate(x_rgb_list[1], scale_factor=4, mode="bilinear", align_corners=True),
                           l_1, l_2, l_3)
            l_1, l_2, l_3 = self.ms3(F.interpolate(x_rgb_list[2], scale_factor=4, mode="bilinear", align_corners=True),
                           l_1, l_2, l_3)
            l_1, l_2, l_3 = self.ms4(F.interpolate(x_rgb, scale_factor=4, mode="bilinear", align_corners=True),
                           l_1, l_2, l_3)
            l_1 = self.uc1(F.interpolate(l_1, scale_factor=2, mode="bilinear", align_corners=True))
            l_2 = self.uc2(F.interpolate(l_2, scale_factor=2, mode="bilinear", align_corners=True))
            l_3 = self.uc3(F.interpolate(l_3, scale_factor=2, mode="bilinear", align_corners=True))

            return x, l_1, l_2, l_3

        return x


if __name__ == '__main__':
    torch.cuda.set_device(0)
    x = torch.rand(4, 3, 256, 256).cuda()
    l1 = torch.rand(4, 3, 256, 256).cuda()
    l2 = torch.rand(4, 3, 256, 256).cuda()
    l3 = torch.rand(4, 3, 256, 256).cuda()
    model = DLM().cuda()
    x, l_1, l_2, l_3 = model(x, l1, l2, l3)
    pass