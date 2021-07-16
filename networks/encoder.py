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
    def __init__(self, out_channels=3):
        super(SCoordNet, self).__init__()
        self.activation = nn.SELU()
        resnet = models.resnet152()
        modules = list(resnet.children())
        self.convs = nn.Sequential(*modules[:8])
        self.mlps = nn.ModuleDict({
            "fc1": nn.Linear(208, 64),
            "fc2": nn.Linear(64, 64),
            "fc3": nn.Linear(64, 32),
            "z": nn.Linear(32, 3),
            "v": nn.Linear(32, 1)
        })

    def forward(self, x):
        feats = self.convs(x)
        feats = feats.flatten(2)
        for i in self.mlps:
            if i == 'z':
                z_t = torch.exp(self.mlps[i](feats)) * 1e-2
            elif i == 'v':
                v_t = self.mlps[i](feats)
            else:
                feats = self.activation(self.mlps[i](feats))

        return v_t, z_t


class OFlowNet(nn.Module):
    def __init__(self, n_channels=1):
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

        self.encoder = nn.ModuleDict({
            "layer1": encoder[:3],
            "layer2": encoder[3:5],
            "layer3": encoder[5:6],
            "layer4": encoder[6:7],
            "layer5": encoder[7:8]
        })
        self.upsample = nn.Upsample(scale_factor=2)
        self.upconvs = nn.ModuleDict({
            "l5_up":
            nn.Conv2d(2048, 2048, 1),
            "l4_up":
            nn.Conv2d(1024, 1024, 1),
            "l3_up":
            nn.Conv2d(512, 512, 1),
            "l2_up":
            nn.Conv2d(256, 256, 1),
            "l1_up":
            nn.Conv2d(64, 64, 1),
            "conv5_up":
            nn.Conv2d(2048 + 1024, 1024, 1, 1),
            "conv4_up":
            nn.Conv2d(1024 + 512, 512, 1, 1),
            "conv3_up":
            nn.Conv2d(512 + 256, 256, 1, 1),
            "conv2_up":
            nn.Conv2d(256 + 64, 64, 1, 1),
            "conv1_up":
            nn.Conv2d(64 + 64, n_channels, 1, 1)
        })
        self.mlps = nn.ModuleDict({
            "fc1": nn.Linear(208, 64),
            "fc2": nn.Linear(64, 64),
            "fc3": nn.Linear(64, 32),
            "out": nn.Linear(32, 1)
        })
        self.feat_extract = nn.ModuleDict({
            "l1": nn.Conv2d(n_channels, 64, 1, 1),
            # "l2": nn.Conv2d(64, 64, 1),
            "l3": nn.Conv2d(64, 32, 1, 1),
            "l4": nn.Conv2d(32, 1, 1),
            "out": nn.Linear(3328, 2048),
            'reshape': nn.Linear(1, 3)
        })
        self.pool = nn.MaxPool2d(2)
        self.conv_img = nn.Conv2d(16, 64, 1, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        img = self.activation(self.conv_img(x))
        # Encoder
        layer1 = self.encoder["layer1"](x)
        layer2 = self.encoder["layer2"](layer1)
        layer3 = self.encoder["layer3"](layer2)
        layer4 = self.encoder["layer4"](layer3)
        layer5 = self.encoder["layer5"](layer4)
        # Encoder Features
        feats = layer5.flatten(2)
        for i in self.mlps:
            if i == 'out':
                feats = torch.exp(self.mlps[i](feats)) * 1e-2
            else:
                feats = self.activation(self.mlps[i](feats))
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

        for i in self.feat_extract:
            if i == 'out':
                x = x.flatten(1)
                probs = self.activation(self.feat_extract[i](x))
                probs = probs.unsqueeze(2)
            elif i == 'reshape':
                probs = self.feat_extract[i](probs)

            else:
                x = self.activation(self.pool(self.feat_extract[i](x)))

        # probs = self.softmax(probs).unsqueeze(2)

        return feats, probs


class PoseEstimator(nn.Module):
    def __init__(self):
        super(PoseEstimator, self).__init__()
        self.activation = nn.SELU()
        self.unflatten = nn.Unflatten(2, (32, 64))
        self.conv = nn.Conv2d(3, 1, 3, 1)
        self.mlps = nn.ModuleDict({
            "fc1": nn.Linear(1860, 1024),
            "fc2": nn.Linear(1024, 512),
            "fc3": nn.Linear(512, 512),
            "fc4": nn.Linear(512, 128),
            "rotation": nn.Linear(128, 9),
            "translation": nn.Linear(128, 3)
        })

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.unflatten(x)
        x = self.activation(self.conv(x))
        x = x.flatten(1)
        for i in self.mlps:
            if i == 'rotation':
                rot = (self.mlps[i](x)).view(-1, 3, 3)
            elif i == 'translation':
                t = (self.mlps[i](x)).unsqueeze(2)
            else:
                x = self.activation(self.mlps[i](x))

        x = torch.cat([rot, t], dim=2)
        return x
