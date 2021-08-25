import os
import torch
from .backbone import *
from torch import nn


class SLAMNet(nn.Module):
    def __init__(
            self, n_channels=1, chkpt_dir="model_checkpoints", model_name="slamnet.pt"
    ):
        super(SLAMNet, self).__init__()
        self.activation = nn.SELU()
        self.encoder = Encoder()
        self.chkpt_dir = chkpt_dir
        self.file = os.path.join(chkpt_dir, model_name)

    def forward(self, img, cam):
        b = img.size()[0]
        h = img.size()[2]
        w = img.size()[3]
        k = cam.inverse()
        img = img.flatten(2)
        img = (k @ img)
        img = img.reshape(b,3,h,w)
        x = self.encoder(img)
        return x

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':
    # model = SLAMNet()
    # x = torch.randn(2, 3, 256, 832)
    # cam = torch.randn(2,3,3)
    # out = model(x, cam)
    # print(out.size())
