import os
from gtrxl_torch.gtrxl_torch import GTrXL
import torch
from .encoder import Encoder
from torch import nn


class PoseNet(nn.Module):
    def __init__(
        self,
        n_layers=6,
        nheads=4,
        model_name='posenet.pt',
        chkpt='model_checkpoints',
    ):
        super(PoseNet, self).__init__()
        self.chkpt_dir = chkpt
        self.file = os.path.join(chkpt, model_name)
        self.actiivation = nn.SELU()
        self.dropout = nn.Dropout(0.4)
        deconvs = {}
        mlps = {}
        convs = {}
        # Set up FC
        self.input_fc = nn.Linear(4180, 512)
        self.translation_fc = nn.Linear(64, 3)
        self.rotation_fc = nn.Linear(64, 3)
        neurons = [512, 64]
        for i in range(len(neurons) - 1):
            layer_name = "fc" + str(i)
            mlps[layer_name] = nn.Linear(neurons[i], neurons[i + 1])
        self.depth_conv = nn.Conv2d(1, 8, 1, 1)
        # Merged Feats Convs
        self.pool = nn.MaxPool2d(2)
        feat_convs = [256, 128, 64, 1]
        for i in range(len(feat_convs) - 1):
            layer_name = "featconvs" + str(i)
            convs[layer_name] = nn.ConvTranspose2d(feat_convs[i],
                                                   feat_convs[i + 1], 3, 1)

        self.unflatten = nn.Unflatten(2, (8, 26))
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.fcl = nn.ModuleDict(mlps)
        self.convs = nn.ModuleDict(convs)
        self.transformer = GTrXL(208,
                                 nheads,
                                 n_layers,
                                 batch_first=True,
                                 activation='gelu')  # Transformer
        self.encoder = Encoder()

    def forward(self, s, s_):
        x = self.encoder(s, s_)
        x = x.flatten(2)
        x = self.transformer(x)
        x = self.unflatten(x)
        x = self.pixel_shuffle(x)
        for i in self.convs:
            x = self.actiivation(self.convs[i](x))
        x = x.flatten(1)

        x = self.actiivation(self.input_fc(x))

        for i in self.fcl:
            x = self.dropout(self.actiivation(self.fcl[i](x)))

        r = self.rotation_fc(x)
        r = self.euler2mat(r)

        t = self.translation_fc(x).unsqueeze(2)
        pose = torch.cat([r, t], dim=2)  # B x 3 x 4

        return pose

    def euler2mat(self, angle):
        """
        Args: [B,3] in radians
        Returns: [B,3,3]

        Reference: https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """

        B = angle.size(0)
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

        cosz = torch.cos(z)
        sinz = torch.sin(z)

        zeros = z.detach() * 0
        ones = zeros.detach() + 1
        zmat = torch.stack(
            [cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones],
            dim=1).reshape(B, 3, 3)

        cosy = torch.cos(y)
        siny = torch.sin(y)

        ymat = torch.stack(
            [cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy],
            dim=1).reshape(B, 3, 3)

        cosx = torch.cos(x)
        sinx = torch.sin(x)

        xmat = torch.stack(
            [ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx],
            dim=1).reshape(B, 3, 3)

        rotMat = xmat @ ymat @ zmat

        return rotMat

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
