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
        self.gru = nn.GRU(208, 32, n_layers, batch_first=True)
        mlps = {}
        layers = [4096, 1024, 512, 256, 32]
        for i in range(len(layers) - 1):
            layer_name = "layer" + str(i)
            mlps[layer_name] = nn.Linear(layers[i], layers[i + 1])

        self.fc = nn.ModuleDict(mlps)
        self.out = nn.Linear(32, 12)
        self.encoder = Encoder()

    def forward(self, s, s_):
        x = self.encoder(s, s_)
        x = x.flatten(2)
        x, h0 = self.gru(x)
        x = x.permute((0, 2, 1))
        for i in self.fc:
            x = self.actiivation(self.fc[i](x))

        x = x.mean(1)
        x = self.out(x).reshape(-1, 3, 4)

        return x

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':
#     ex = torch.randn(7, 3, 832, 256)
#     model = PoseNet()
#     y = model(ex, ex)
#     print(y.size())
