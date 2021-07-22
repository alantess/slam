import os
import torch
from .encoder import Encoder
from torch import nn


class DepthNet(nn.Module):
    def __init__(self,
                 n_channels=1,
                 chkpt_dir="model_checkpoints",
                 model_name="depthnet152.pt"):
        super(DepthNet, self).__init__()
        channels = [4096, 2048, 1024, 512, 256, 128]
        deconvs = {}
        self.activation = nn.SELU()
        for i in range(len(channels) - 1):
            layer_name = "layer" + str(i)
            deconvs[layer_name] = nn.ConvTranspose2d(channels[i],
                                                     channels[i + 1], 2, 2)

        self.encoder = Encoder()
        self.decoder = nn.ModuleDict(deconvs)
        self.output_layer_1 = nn.ConvTranspose2d(256, 128, 1, 1)
        self.output_layer_2 = nn.ConvTranspose2d(128, 64, 1, 1)
        self.output_layer_3 = nn.ConvTranspose2d(64, 16, 1, 1)
        self.output_layer_4 = nn.ConvTranspose2d(32, 1, 1, 1)

        self.chkpt_dir = chkpt_dir
        self.file = os.path.join(chkpt_dir, model_name)
        self.conv = nn.Conv2d(3, 64, 1, 1)
        self.conv_s_ = nn.Conv2d(3, 16, 1, 1)

    def forward(self, s, s_):
        start_frame = self.activation(self.conv(s))
        next_frame = self.activation(self.conv(s_))
        decoded_frame = self.activation(self.conv_s_(s_))
        original = torch.cat([start_frame, next_frame], dim=1)

        x = self.encoder(s, s_)
        for i in self.decoder:
            x = self.activation(self.decoder[i](x))

        x = torch.cat([x, original], dim=1)

        x = self.activation(self.output_layer_1(x))
        x = self.activation(self.output_layer_2(x))
        x = self.activation(self.output_layer_3(x))
        x = torch.cat([x, decoded_frame], dim=1)
        x = self.output_layer_4(x)

        return x

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':
#     model = DepthNet()
#     x = torch.randn(1, 3, 256, 832)
#     out = model(x, x)
# print(out.size())
