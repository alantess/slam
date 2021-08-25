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
        self.activation = nn.SELU()
        self.gru = nn.GRU(size, size * 2, layers, batch_first=True)
        self.h_layer1 = nn.Linear(size * 2, size * 2)
        self.h_layer2 = nn.Linear(size * 2, size * 2)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(size * 2, int(size * 1.5))

    def forward(self, img):
        b = img.size(0)
        x = img.flatten(2)
        x, h0 = self.gru(x)
        x = self.activation(self.h_layer1(x))
        x = self.activation(self.drop(self.h_layer2(x)))
        x = self.out(x)
        x = x.flatten(1)
        feats = x.size(1) // 3
        x = x.reshape(b, feats, 3)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        deconvs = {}
        channels = [2048,1024,512,256,128, 3]
        for c in range(len(channels) - 1):
            name = "conv" + str(c)
            deconvs[name] = nn.ConvTranspose2d(channels[c], channels[c+1], 2,2)

        self.deconvs = nn.ModuleDict(deconvs)
        self.activation = nn.SELU()


    def forward(self, x):
        for i in self.deconvs:
            x = self.activation(self.deconvs[i](x))
        return x


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.activation = nn.SELU()
        layers = [6,32,128,128,32,16,1]
        mlp = {}
        for i in range(len(layers) -1):
            name = "layer" + str(i)
            mlp[name] = nn.Linear(layers[i] , layers[i+1])
        self.mlp = nn.ModuleDict(mlp)


        
    def forward(self, x,y):
        x = torch.cat((x,y),1)
        x = x.flatten(2).permute(0,2,1)
        for i in self.mlp:
            if i == 'layer5':
                x = self.mlp[i](x)
            else:
                x = self.activation(self.mlp[i](x))
        return x

    
