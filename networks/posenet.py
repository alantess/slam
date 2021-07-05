import os
import torch
from torch import nn
from torchvision import models


class PoseDecoder(nn.Module):
    def __init__(self,
                 enc_channels,
                 num_input_features=1,
                 stride=1,
                 num_frames=2):
        super(PoseDecoder, self).__init__()
        self.activation = nn.SELU()
        self.convs = {}
        self.convs[("squeeze")] = nn.Conv2d(enc_channels, 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3,
                                            stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 256 * num_frames, 1)
        self.fc1 = nn.Linear(256 * num_frames, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 12)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, feats):
        features = [f[-1] for f in feats]
        cat_features = [
            self.activation(self.convs["squeeze"](f)) for f in features
        ]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features

        for i in range(3):
            out = self.convs[('pose', i)](out)
            if i != 2:
                out = self.activation(out)

        pose = out.flatten(1)
        pose = self.activation(self.fc1(pose))
        pose = self.activation(self.fc2(pose))
        pose = self.activation(self.fc3(pose))
        pose = (self.out(pose)).reshape(-1, 3, 4)

        return pose


class PoseNet(nn.Module):
    def __init__(self, model_name='posenet.pt', chkpt='model_checkpoints'):
        super(PoseNet, self).__init__()
        self.chkpt_dir = chkpt
        self.file = os.path.join(chkpt, model_name)
        self.convs_reset = nn.Conv2d(6, 3, 1, 1)
        resnet = models.resnet152(True)
        modules = list(resnet.children())
        self.encoder = nn.Sequential(*modules[:9])

        self.decoder = PoseDecoder(4096)

    def forward(self, img_a, img_b):
        src = self.encoder(img_a).unsqueeze(0)
        tgt = self.encoder(img_b).unsqueeze(0)
        feats = torch.cat([src, tgt], dim=2)
        pose = self.decoder([feats])
        return pose

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':
#     ex = torch.randn(4, 3, 256, 256)
#     model = PoseNet()
#     y = model(ex, ex)
#     print(y.size())
