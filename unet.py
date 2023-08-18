import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv2d(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DoubleConv2d, self).__init__()
        self.double_conv2d = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv2d(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.double_conv1 = DoubleConv2d(1, 64)
        self.double_conv2 = DoubleConv2d(64, 128)
        self.double_conv3 = DoubleConv2d(128, 256)
        self.double_conv4 = DoubleConv2d(256, 512)
        self.double_conv5 = DoubleConv2d(512, 1024)
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_conv6 = DoubleConv2d(1024, 512)
        self.double_conv7 = DoubleConv2d(512, 256)
        self.double_conv8 = DoubleConv2d(256, 128)
        self.double_conv9 = DoubleConv2d(128, 64)
        self.conv_1x1 = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.double_conv1(x)
        x2 = self.max_pool_2x2(x1)
        x3 = self.double_conv2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.double_conv3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.double_conv4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.double_conv5(x8)

        # decoder
        y = self.up_conv1(x9)
        x7 = TF.resize(x7, y.shape[2])
        y = torch.concat((y, x7), dim=1)
        y = self.double_conv6(y)
        y = self.up_conv2(y)
        x5 = TF.resize(x5, y.shape[2])
        y = torch.concat((y, x5), dim=1)
        y = self.double_conv7(y)
        y = self.up_conv3(y)
        x3 = TF.resize(x3, y.shape[2])
        y = torch.concat((y, x3), dim=1)
        y = self.double_conv8(y)
        y = self.up_conv4(y)
        x1 = TF.resize(x1, y.shape[2])
        y = torch.concat((y, x1), dim=1)
        y = self.double_conv9(y)
        y = self.conv_1x1(y)
        return y
