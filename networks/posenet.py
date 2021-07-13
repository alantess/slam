import os
import torch
from .encoder import *
from torch import nn


# Inputs are 3xTxHxW
class PoseNet3D(nn.Module):
    def __init__(self, model_name="posenet3d.pt", chkpt="model_checkpoints"):
        super(PoseNet3D, self).__init__()
        self.chkpt_dir = chkpt
        self.file = os.path.join(chkpt, model_name)
        self.actiivation = nn.GELU()
        self.encoder = Encoder3d()
        mlps = {}
        fc_neurons = [512, 256, 256, 128, 64, 32]
        for i in range(len(fc_neurons) - 1):
            layer_name = "fc" + str(i)
            mlps[layer_name] = nn.Linear(fc_neurons[i], fc_neurons[i + 1])
        mlps["translation"] = nn.Linear(32, 3)
        mlps["rotation"] = nn.Linear(32, 9)
        self.fcl = nn.ModuleDict(mlps)

    def forward(self, x):
        x = self.encoder(x).flatten(1)
        for i in (self.fcl):
            if i == 'translation':
                translation = (self.fcl[i](x)).unsqueeze(2)
            elif i == 'rotation':
                rotation = (self.fcl[i](x)).view(-1, 3, 3)
            else:
                x = self.actiivation(self.fcl[i](x))
        pose = torch.cat([translation, rotation], dim=2)

        return pose

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':
#     torch.manual_seed(55)
#     ex = torch.randn(1, 3, 16, 256, 832)
#     k = torch.randn(1, 3, 3)
#     model = PoseNet3D()
#     y = model(ex)
#     print(y.size())

# #     model = PoseNet()
# #     y = model(ex, k)
# #     print(y)
# #     print(y.size())
