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
    def __init__(self, size_h, layers):
        super(CalibNet, self).__init__()
        self.rnn = nn.GRU(3, size_h, layers, batch_first=True)
        self.out = nn.Linear(size_h,3)

    def forward(self, img, calib):
        uv = self.pixel_to_cam(img, calib)
        uv, h0 = self.rnn(uv)
        uv = self.out(uv)
        return uv

    def set_id_grid(self, depth: Tensor):
        b, h, w = depth.size()
        # Traverse though the X and Y of the pixel
        i_range = (
            torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)
        )  # 1xHxW
        j_range = (
            torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)
        )  # 1xHxW
        ones = torch.ones(1, h, w).type_as(depth)
        return torch.stack((j_range, i_range, ones), dim=1)

    def pixel_to_cam(self, img: Tensor, K: Tensor):
        """
        Converts a depth map into pixel coordinates
        s[u,v,1] = k[Xc,Yc,Zc] --> [u,v,1] = 1/s[Xc,Yc,Zc]
        Args:
            img [B,H,W]
            K [3,3]
        Returns:
            (Nx3) Perspective Projection
        """
        if int(len(img.size())) == 4:
            img = img.squeeze(1)

        b, h, w = img.size()
        img = 1 / img
        inv = K.inverse()

        pixel_coords = self.set_id_grid(img)

        current_pixel_coords = (
            pixel_coords[:, :, :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)
        )  # [B,3,HxW]

        cam_coords = (inv @ current_pixel_coords).reshape(b, 3, h, w)
        uv = cam_coords * img.unsqueeze(1)
        uv = uv.flatten(2).permute(0, 2, 1)  # BxNx3
        return uv
