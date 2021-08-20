import os
import torch
from .backbone import *
from torch import nn


class DepthNet(nn.Module):
    def __init__(self,
                 n_channels=1,
                 chkpt_dir="model_checkpoints",
                 model_name="depthnet152.pt"):
        super(DepthNet, self).__init__()
        channels = [2048, 1024, 512, 256, 128]
        deconvs = {}
        self.activation = nn.SELU()
        for i in range(len(channels) - 1):
            layer_name = "layer" + str(i)
            deconvs[layer_name] = nn.ConvTranspose2d(channels[i],
                                                     channels[i + 1], 2, 2)

        self.encoder = Encoder()
        self.decoder = nn.ModuleDict(deconvs)
        self.output_layer_1 = nn.ConvTranspose2d(128, 128, 1, 1)
        self.output_layer_2 = nn.ConvTranspose2d(128, 64, 1, 1)
        self.output_layer_3 = nn.ConvTranspose2d(64, 16, 2, 2)
        self.output_layer_4 = nn.ConvTranspose2d(16, 1, 1, 1)

        self.chkpt_dir = chkpt_dir
        self.file = os.path.join(chkpt_dir, model_name)
        self.conv = nn.Conv2d(3, 64, 1, 1)
        self.conv_s_ = nn.Conv2d(3, 16, 1, 1)
        self.calib = CalibNet(10,2)

    def forward(self, img,cam):

        x = self.encoder(img)
        for i in self.decoder:
            x = self.activation(self.decoder[i](x))


        x = self.activation(self.output_layer_1(x))
        x = self.activation(self.output_layer_2(x))
        x = self.activation(self.output_layer_3(x))
        x = self.output_layer_4(x)
        x = self.calib(x, cam)

        return x

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':
#     model = DepthNet()
#     x = torch.randn(2, 3, 256, 832)
#     cam = torch.randn(2,3,3)
#     out = model(x, cam)
#     print(out.size())
