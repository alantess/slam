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
        self.gru = nn.GRU(size, size * 2, layers, batch_first=True)
        self.out = nn.Linear(size * 2, int(size * 0.5))

    def forward(self, img):
        b = img.size(0)
        x = img.flatten(2)
        x, h0 = self.gru(x)
        x = self.get_uv(x)
        x = self.out(x)
        x = x.flatten(2)
        feats = x.size(2)
        x = x.reshape(b, feats, 3)
        return x

    def set_id_grid(self, x: Tensor):
        b, h, w = x.size()
        # Traverse though the X and Y of the pixel
        i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(x)  # 1xHxW
        j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(x)  # 1xHxW
        ones = torch.ones(1, h, w).type_as(x)
        return torch.stack((j_range, i_range, ones), dim=1)

    def get_uv(self, x: Tensor):
        if int(len(x.size())) == 4:
            x = x.squeeze(1)

        b, h, w = x.size()
        x = 1 / x
        pixel_coords = self.set_id_grid(x)
        current_pixel_coords = (
            pixel_coords[:, :, :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)
        )

        cam_coords = current_pixel_coords.reshape(b, 3, h, w)
        uv = cam_coords * x.unsqueeze(1)

        return uv
