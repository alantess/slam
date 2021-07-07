import os
import torch
from .encoder import Encoder
from torch import nn
"""

Takes in a concatenated image (st,st+1)
outputs the depth map of that image
Use the focal loss and smooth l1 loss
"""


class DepthNet(nn.Module):
    def __init__(self,
                 n_channels=1,
                 chkpt_dir="model_checkpoints",
                 model_name="depthnet.pt"):
        super(DepthNet, self).__init__()
        channels = [4096, 2048, 1024, 512, 256, 128]
        convs = {}
        self.activation = nn.SELU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        for i in range(len(channels) - 1):
            layer_name = "layer" + str(i)
            convs[layer_name] = nn.Conv2d(channels[i], channels[i + 1], 1, 1)

        self.encoder = Encoder()
        self.decoder = nn.ModuleDict(convs)
        self.output_layer_1 = nn.Conv2d(128, 64, 1, 1)
        self.output_layer_2 = nn.Conv2d(64, 3, 1, 1)

        self.chkpt_dir = chkpt_dir
        self.file = os.path.join(chkpt_dir, model_name)

    def forward(self, s, s_):
        x = self.encoder(s, s_)
        for i in self.decoder:
            x = self.upsample(x)
            x = self.activation(self.decoder[i](x))

        x = self.activation(self.output_layer_1(x))
        x = self.output_layer_2(x)

        return x

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':
#     model = DepthNet()
#     x = torch.randn(1, 3, 832, 256)
#     out = model(x, x)
#     print(out.size())
