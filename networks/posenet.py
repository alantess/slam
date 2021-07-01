import os
import torch
from collection import OrderedDict
from torch import nn
from torchvision import models


class PoseDecoder(nn.Module):
    def __init__(self,
                 enc_channels,
                 num_input_features=1,
                 stride=1,
                 num_frames=1):
        super(PoseDecoder, self).__init__()
        self.activation = nn.LeakyReLU()
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(enc_channels, 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3,
                                            stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames, 1)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, feats):
        features = [f[-1] for f in feats]
        cat_features = [
            self.activation(self.convs["squeeze"](f)) for f in features
        ]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features

        for i in range(3):
            out = self.convs[('pose', i)](out)
            if i != 2:
                out = self.activation(out)

        out = out.mean(3).mean(2)

        pose = 0.01 * out.view(-1, 6)
        return pose


class PoseNet(nn.Module):
    def __init__(self, model_name='posenet.pt', chkpt='model_checkpoints'):
        super(PoseNet, self).__init__()
        self.encoder = models.resnet152(True)
        self.decoder = PoseDecoder(2048)

    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        feats = self.encoder(x)
        pose = self.decoder([feats])
        return pose