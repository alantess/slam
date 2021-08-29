import torch
from torch import nn, Tensor
from torchvision import models


class Encoder(nn.Module):
    # resnet-152 encoder for the pose net and depth network
    # takes in 2 images (st-1,st)
    # outputs the concatenated images between the two

    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet152()
        modules = list(resnet.children())
        # Original Image
        self.original_conv1 = nn.Conv2d(3, 64, 1, 1)
        self.original_conv2 = nn.Conv2d(64, 64, 1, 1)
        # Encoder
        self.layer1 = nn.Sequential(*modules[:3])
        self.layer2 = nn.Sequential(*modules[3:5])
        self.layer3 = nn.Sequential(*modules[5:6])
        self.layer4 = nn.Sequential(*modules[6:7])
        self.layer5 = nn.Sequential(*modules[7:8])
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer5_up = nn.Conv2d(2048, 2048, 1)
        self.layer4_up = nn.Conv2d(1024, 1024, 1)
        self.layer3_up = nn.Conv2d(512, 512, 1)
        self.layer2_up = nn.Conv2d(256, 256, 1)
        self.layer1_up = nn.Conv2d(64, 64, 1)

        # Reduce the channels
        self.conv5_up = nn.Conv2d(2048 + 1024, 1024, 1, 1)
        self.conv4_up = nn.Conv2d(1024 + 512, 512, 1, 1)
        self.conv3_up = nn.Conv2d(512 + 256, 256, 1, 1)
        self.conv2_up = nn.Conv2d(256 + 64, 64, 1, 1)
        self.conv1_up = nn.Conv2d(64 + 64, 1, 1, 1)
        self.calib = CalibNet(208,4)
        self.activation = nn.SELU()


    def forward(self, x):
        original = self.activation(self.original_conv1(x))
        original = self.activation(self.original_conv2(original))
        # Encoder
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer5 = self.layer5_up(layer5)
        b = layer5.size(0)
        d = layer5.size(1)
        h = layer5.size(2)
        w = layer5.size(3)

        layer5 = self.calib(layer5).reshape(b,d,h,w)
        # Decoder L5
        x = self.upsample(layer5)
        layer4 = self.activation(self.layer4_up(layer4))
        x = torch.cat([x, layer4], dim=1)
        x = self.activation(self.conv5_up(x))
        # L4
        x = self.upsample(x)
        layer3 = self.activation(self.layer3_up(layer3))
        x = torch.cat([x, layer3], dim=1)
        x = self.activation(self.conv4_up(x))
        # L3
        x = self.upsample(x)
        layer2 = self.activation(self.layer2_up(layer2))
        x = torch.cat([x, layer2], dim=1)
        x = self.activation(self.conv3_up(x))
        # L2
        x = self.upsample(x)
        layer1 = self.activation(self.layer1_up(layer1))
        x = torch.cat([x, layer1], dim=1)
        x = self.activation(self.conv2_up(x))
        # L1
        x = self.upsample(x)
        x = torch.cat([x, original], dim=1)
        x = self.conv1_up(x)

        return x


class CalibNet(nn.Module):
    def __init__(self, size, layers):
        super(CalibNet, self).__init__()
        self.activation = nn.SELU()
        encoder_layer = nn.TransformerEncoderLayer(d_model=208, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)

    def forward(self, img):
        x = img.flatten(2)
        x = self.transformer(x)
        return x

