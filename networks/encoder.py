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
