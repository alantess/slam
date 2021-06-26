import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models


class DisparityNet(nn.Module):
    """
    Resnet50 U-Net Module
    """
    def __init__(self, n_out_channels=1, chkpt_dir="model_checkpoints"):
        super(DisparityNet, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.file = os.path.join(chkpt_dir, "disparity_net.pt")
        resnet = models.resnet50(True)
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
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)
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
        self.conv1_up = nn.Conv2d(64 + 64, n_out_channels, 1, 1)

    def forward(self, x):
        original = F.selu(self.original_conv1(x))
        original = F.selu(self.original_conv2(original))
        # Encoder
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer5 = self.layer5_up(layer5)
        # Decoder L5
        x = self.upsample(layer5)
        layer4 = F.selu(self.layer4_up(layer4))
        x = torch.cat([x, layer4], dim=1)
        x = F.selu(self.conv5_up(x))
        # L4
        x = self.upsample(x)
        layer3 = F.selu(self.layer3_up(layer3))
        x = torch.cat([x, layer3], dim=1)
        x = F.selu(self.conv4_up(x))
        # L3
        x = self.upsample(x)
        layer2 = F.selu(self.layer2_up(layer2))
        x = torch.cat([x, layer2], dim=1)
        x = F.selu(self.conv3_up(x))
        # L2
        x = self.upsample(x)
        layer1 = F.selu(self.layer1_up(layer1))
        x = torch.cat([x, layer1], dim=1)
        x = F.selu(self.conv2_up(x))
        # L1
        x = self.upsample(x)
        x = torch.cat([x, original], dim=1)
        x = self.conv1_up(x)

        x = torch.sigmoid(x)
        return x

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))
