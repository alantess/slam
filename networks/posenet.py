import os
import torch
from .encoder import Encoder
from torch import nn


class PoseNet(nn.Module):
    def __init__(
        self,
        n_layers=4,
        model_name='posenet.pt',
        chkpt='model_checkpoints',
    ):
        super(PoseNet, self).__init__()
        self.chkpt_dir = chkpt
        self.file = os.path.join(chkpt, model_name)
        self.actiivation = nn.SELU()
        self.dropout = nn.Dropout(0.4)
        mlps = {}
        convs = {}
        self.gru = nn.GRU(2548, 512, n_layers, batch_first=True)
        # Set up FC
        self.input_fc = nn.Linear(512, 256)
        self.translation_fc = nn.Linear(32, 3)
        self.rotation_fc = nn.Linear(32, 9)
        neurons = [256, 128, 32]
        for i in range(len(neurons) - 1):
            layer_name = "fc" + str(i)
            mlps[layer_name] = nn.Linear(neurons[i], neurons[i + 1])

        layers = [256, 128, 64, 1]
        for i in range(len(layers) - 1):
            layer_name = "layer" + str(i)
            convs[layer_name] = nn.Conv2d(layers[i], layers[i + 1], 3, 1)

        self.pixel_shuffle = nn.PixelShuffle(4)
        self.fcl = nn.ModuleDict(mlps)
        self.convs = nn.ModuleDict(convs)
        self.encoder = Encoder()

    def forward(self, s, s_):
        x = self.encoder(s, s_)
        x = self.pixel_shuffle(x)
        for i in self.convs:
            x = self.actiivation(self.convs[i](x))

        x = x.flatten(2)
        x, _ = self.gru(x)
        x = x.mean(1)

        x = self.actiivation(self.input_fc(x))

        for i in self.fcl:
            x = self.dropout(self.actiivation(self.fcl[i](x)))

        r = self.rotation_fc(x).view(-1, 3, 3)

        t = self.translation_fc(x).unsqueeze(2)
        pose = torch.cat([r, t], dim=2)  # B x 3 x 4

        return pose

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':

#     ex = torch.randn(1, 3, 256, 832)

#     model = PoseNet()
#     y = model(ex, ex)
#     print(y.size())
