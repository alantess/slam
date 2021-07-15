import torch
from torch import nn
from torchvision import models


class Encoder(nn.Module):
    # resnet-152 encoder for the pose net and depth network
    # takes in 2 images (st-1,st)
    # outputs the concatenated images between the two

    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet152(True)
        modules = list(resnet.children())
        self.encoder = nn.Sequential(*modules[:8])

    def forward(self, src, tgt):
        src = self.encoder(src)
        tgt = self.encoder(tgt)
        encoded = torch.cat([src, tgt], dim=1)  # Bx4096xHxW
        return encoded


class SCoordNet(nn.Module):
    def __init__(self, out_channels=4):
        super(SCoordNet, self).__init__()
        convs = {}
        self.activation = nn.SELU()
        convolutions = [[3, 3, 1], [3, 64, 1], [3, 64, 1], [3, 256, 2],
                        [3, 256, 1], [3, 512, 2], [3, 512, 1], [3, 1024, 2],
                        [3, 1024, 1], [3, 512, 1], [3, 256, 1], [1, 128, 1],
                        [1, out_channels, 1]]

        for i in range(len(convolutions) - 1):
            layer_name = 'conv' + str(i)
            kernel = convolutions[i][0]
            input_channel = convolutions[i][1]
            out_channel = convolutions[i + 1][1]
            stride = convolutions[i][2]

            convs[layer_name] = nn.Conv2d(input_channel,
                                          out_channel,
                                          kernel,
                                          stride=stride)

        self.convs = nn.ModuleDict(convs)

    def forward(self, x):
        for i in self.convs:
            if i == 'conv11':
                x = self.convs[i](x)
            else:
                x = self.activation(self.convs[i](x))

        return x


class OFlowNet(nn.Module):
    def __init__(self, n_channels=3):
        super(OFlowNet, self).__init__()
        self.activation = nn.SELU()
        resnet = models.resnet152()
        modules = list(resnet.children())
        encoder = nn.Sequential(*modules[:8])
        encoder[0] = nn.Conv2d(16,
                               64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=(3, 3),
                               bias=False)

        self.encoder = {
            "layer1": encoder[:3],
            "layer2": encoder[3:5],
            "layer3": encoder[5:6],
            "layer4": encoder[6:7],
            "layer5": encoder[7:8]
        }
        self.upsample = nn.Upsample(scale_factor=2)
        decoder = {
            "l5_up": nn.Conv2d(2048, 2048, 1),
            "l4_up": nn.Conv2d(1024, 1024, 1),
            "l3_up": nn.Conv2d(512, 512, 1),
            "l2_up": nn.Conv2d(256, 256, 1),
            "l1_up": nn.Conv2d(64, 64, 1),
            "conv5_up": nn.Conv2d(2048 + 1024, 1024, 1, 1),
            "conv4_up": nn.Conv2d(1024 + 512, 512, 1, 1),
            "conv3_up": nn.Conv2d(512 + 256, 256, 1, 1),
            "conv2_up": nn.Conv2d(256 + 64, 64, 1, 1),
            "conv1_up": nn.Conv2d(64 + 64, n_channels, 1, 1)
        }
        self.conv_img = nn.Conv2d(16, 64, 1, 1)
        self.upconvs = nn.ModuleDict(decoder)

    def forward(self, x):
        img = self.activation(self.conv_img(x))
        # Encoder
        layer1 = self.encoder["layer1"](x)
        layer2 = self.encoder["layer2"](layer1)
        layer3 = self.encoder["layer3"](layer2)
        layer4 = self.encoder["layer4"](layer3)
        layer5 = self.encoder["layer5"](layer4)
        # L5
        x = self.upsample(layer5)
        layer4 = self.activation(self.upconvs["l4_up"](layer4))
        x = torch.cat([x, layer4], dim=1)
        x = self.activation(self.upconvs["conv5_up"](x))
        # L4
        x = self.upsample(x)
        layer3 = self.activation(self.upconvs["l3_up"](layer3))
        x = torch.cat([x, layer3], dim=1)
        x = self.activation(self.upconvs["conv4_up"](x))
        # L3
        x = self.upsample(x)
        layer2 = self.activation(self.upconvs["l2_up"](layer2))
        x = torch.cat([x, layer2], dim=1)
        x = self.activation(self.upconvs["conv3_up"](x))
        # L2
        x = self.upsample(x)
        layer1 = self.activation(self.upconvs["l1_up"](layer1))
        x = torch.cat([x, layer1], dim=1)
        x = self.activation(self.upconvs["conv2_up"](x))
        # L1
        x = self.upsample(x)
        x = torch.cat([x, img], dim=1)
        x = self.upconvs["conv1_up"](x)
        return x
