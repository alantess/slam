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
        self.encoder = nn.Sequential(*modules[:8])

    def forward(self, src):
        src = self.encoder(src)
        return src


class CalibNet(nn.Module):
    def __init__(self, size, layers):
        super(CalibNet, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=size, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers * 2)
        self.gru = nn.GRU(size, int(size * 1.5), layers, batch_first=True)

    def forward(self, img):
        b = img.size(0)
        img = img.flatten(2)
        x = self.transformer(img)
        x, h0 = self.gru(x)
        x = x.flatten(1)
        feats = x.size(1) // 3
        x = x.reshape(b, feats, 3)
        return x
