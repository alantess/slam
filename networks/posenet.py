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
        self.actiivation = nn.GELU()
        mlps = {}
        convs = {}
        # Set up FC
        self.input_fc = nn.Linear(11264, 512)
        self.translation_fc = nn.Linear(32, 3)
        self.rotation_fc = nn.Linear(32, 9)
        neurons = [512, 128, 128, 32]
        for i in range(len(neurons) - 1):
            layer_name = "fc" + str(i)
            mlps[layer_name] = nn.Linear(neurons[i], neurons[i + 1])
        # Convs
        layers = [3, 16, 32, 64, 128, 256, 512]
        for i in range(len(layers) - 1):
            layer_name = "layer" + str(i)
            convs[layer_name] = nn.Conv2d(layers[i], layers[i + 1], 3, 1)

        self.pool = nn.MaxPool2d(2)
        self.fcl = nn.ModuleDict(mlps)
        self.convs = nn.ModuleDict(convs)

    def forward(self, depth, k_inv):
        depth = depth.squeeze(1)
        x = self.get_pixels(depth, k_inv)
        for i in self.convs:
            x = self.actiivation(self.pool(self.convs[i](x)))

        x = x.flatten(1)
        x = self.actiivation(self.input_fc(x))
        for i in self.fcl:
            x = self.actiivation(self.fcl[i](x))

        r = self.rotation_fc(x).view(-1, 3, 3)
        t = self.translation_fc(x).unsqueeze(2)
        x = torch.cat([r, t], dim=2)
        return x

    def get_pixels(self, depth, intrinsics_inv):
        b, h, w = depth.size()
        pixel_coords = self.set_id_grid(depth)
        current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
            b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
        cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(
            b, 3, h, w)

        return cam_coords * depth.unsqueeze(1)

    def set_id_grid(self, depth):
        b, h, w = depth.size()
        i_range = torch.arange(0,
                               h).view(1, h,
                                       1).expand(1, h,
                                                 w).type_as(depth)  # [1, H, W]
        j_range = torch.arange(0,
                               w).view(1, 1,
                                       w).expand(1, h,
                                                 w).type_as(depth)  # [1, H, W]
        ones = torch.ones(1, h, w).type_as(depth)

        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)
        return pixel_coords

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':

#     torch.manual_seed(55)
#     ex = torch.randn(1, 1, 256, 832)
#     k = torch.randn(1, 3, 3)

#     model = PoseNet()
#     y = model(ex, k, k)
#     print(y)
#     print(y.size())
