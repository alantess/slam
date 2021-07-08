import os
import torch
from .encoder import Encoder
from torch import nn


class PoseNet(nn.Module):
    def __init__(
        self,
        n_layers=2,
        model_name='posenet.pt',
        chkpt='model_checkpoints',
    ):
        super(PoseNet, self).__init__()
        self.chkpt_dir = chkpt
        self.file = os.path.join(chkpt, model_name)
        self.actiivation = nn.SELU()
        # Set up RNN
        self.gru = nn.GRU(212992, 1024, n_layers, batch_first=True)
        self.gru_original = nn.GRU(53248, 1024, n_layers, batch_first=True)
        deconvs = {}
        mlps = {}
        neurons = [1024, 512, 256, 128, 32]
        # Set up FC
        for i in range(len(neurons) - 1):
            layer_name = "fc" + str(i)
            mlps[layer_name] = nn.Linear(neurons[i], neurons[i + 1])
        # Set up Deconvs
        layers = [4096, 2048, 1024, 512, 256, 32]
        for i in range(len(layers) - 1):
            layer_name = "layer" + str(i)
            deconvs[layer_name] = nn.ConvTranspose2d(layers[i], layers[i + 1],
                                                     2, 2)
        self.conv_og = nn.Conv2d(3, 32, 2, 2)

        self.decoder = nn.ModuleDict(deconvs)
        self.fcl = nn.ModuleDict(mlps)
        self.out = nn.Linear(32, 12)
        self.encoder = Encoder()

    def forward(self, s, s_):
        x = self.encoder(s, s_)
        s = self.actiivation(self.conv_og(s)).flatten(2)
        s_ = self.actiivation(self.conv_og(s_)).flatten(2)
        s0, h0 = self.gru_original(s)
        s1, h1 = self.gru_original(s_, h0)

        for i in self.decoder:
            x = self.actiivation(self.decoder[i](x))

        x = x.flatten(2)
        x, _ = self.gru(x, h1)
        x = x.mean(1)
        for i in self.fcl:
            x = self.actiivation(self.fcl[i](x))

        x = self.out(x).view(-1, 3, 4)

        return x

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':
# ex = torch.randn(7, 3, 256, 832)
# model = PoseNet()
# y = model(ex, ex)
# print(y.size())
