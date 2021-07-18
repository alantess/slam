import os
import torch
from .encoder import *
from torch import nn


# Reference:  https://arxiv.org/pdf/2003.10629v1.pdf (KFNet: Learning Temporal Camera Relocalization using Kalman Filtering)
class KFNet(nn.Module):
    def __init__(
        self,
        model_name='kfnet.pt',
        chkpt='model_checkpoints',
    ):
        super(KFNet, self).__init__()
        self.chkpt_dir = chkpt
        self.file = os.path.join(chkpt, model_name)
        self.activation = nn.SELU()
        # Input Convs
        self.cost_volume = nn.Conv2d(3, 8, 1, 1)
        # SCoordNet (Measurement Systems)
        self.scoord = SCoordNet()
        # OFlow (Process Systems)
        self.o_flow = OFlowNet()
        # Pose estimation
        self.pose_estimator = PoseEstimator()

        # Filtering System
    def forward(self, prev, cur):
        # State obs and measurement noise covariance
        v_t, z_t = self.scoord(cur)  #Bx2048x3 & Bx2048x1
        cur = self.cost_volume(cur)
        prev = self.cost_volume(prev)
        encode = torch.cat([prev, cur], dim=1)
        # Prior noise covariance and prior state mean
        w_t, g_t = self.o_flow(encode)  #Bx2048x3 & Bx2048x1

        prior_state_mean = self.activation(g_t)
        prior_state_covar = self.activation(g_t) + w_t  # R_t

        mean_t, covar_t = self.kalman_filter(prior_state_mean,
                                             prior_state_covar, z_t, v_t)

        pose = self.pose_estimator(mean_t)

        return pose

    def kalman_filter(self, prev_mean, r_t, z_t, v_t):
        estimated_mean = z_t - prev_mean  # innovation
        k_t = r_t / (r_t + v_t)  # Kalman Gain
        new_state = prev_mean + (k_t * estimated_mean)  # Update state mean
        new_r_t = r_t * (1 - k_t)  # Update state covariance
        return new_state, new_r_t

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


class PoseNet(nn.Module):
    def __init__(self, model_name="posenetrnn.pt", chkpt="model_checkpoints"):
        super(PoseNet, self).__init__()
        self.chkpt_dir = chkpt
        self.file = os.path.join(chkpt, model_name)
        self.activation = nn.SELU()
        self.encoder = Encoder()
        self.mlps = nn.ModuleDict({
            "fc1": nn.Linear(2485, 256),
            "fc2": nn.Linear(256, 512),
            "fc3": nn.Linear(512, 512),
        })
        self.prev_mlps = nn.ModuleDict({
            "fc1": nn.Linear(768, 512),
            "fc2": nn.Linear(512, 256),
            "fc3": nn.Linear(256, 256),
            "fc4": nn.Linear(256, 128),
            "translation": nn.Linear(128, 3),
            "rot": nn.Linear(128, 9)
        })
        self.convs = nn.ModuleDict({
            "conv1a":
            nn.ConvTranspose2d(4096, 2048, 3, 1),
            "conv1b":
            nn.ConvTranspose2d(2048, 2048, 1, 1),
            "conv2a":
            nn.ConvTranspose2d(2048, 1048, 3, 1),
            "conv2b":
            nn.ConvTranspose2d(1048, 1048, 1, 1),
            "conv3a":
            nn.ConvTranspose2d(1048, 512, 3, 1),
            "conv3b":
            nn.ConvTranspose2d(512, 512, 1, 1),
            "conv4a":
            nn.ConvTranspose2d(512, 256, 3, 1),
            "conv4b":
            nn.ConvTranspose2d(256, 256, 1, 1),
            "conv5":
            nn.ConvTranspose2d(256, 128, 3, 2),
            "conv6":
            nn.ConvTranspose2d(128, 64, 3, 1),
            "conv7":
            nn.ConvTranspose2d(64, 3, 1, 1),
        })
        self.prev_conv = nn.ModuleDict({
            "conv1a": nn.Conv2d(3, 16, 3, 1),
            "conv1b": nn.Conv2d(16, 32, 1, 2),
            "conv2a": nn.Conv2d(32, 64, 3, 1),
            "conv2b": nn.Conv2d(64, 128, 1, 2),
            "conv3a": nn.Conv2d(128, 256, 1, 1),
            "conv3b": nn.Conv2d(256, 256, 1, 1),
        })

        self.decode_fc = nn.Linear(256, 128)
        self.rot_fc = nn.Linear(128, 9)
        self.transl_fc = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2)
        self.gru = nn.GRU(512, 256, 5, batch_first=True)

    def forward(self, prev, cur):
        x = self.encoder(prev, cur)
        for i in self.convs:
            x = self.activation(self.convs[i](x))
        x = x.flatten(2)
        for i in self.prev_conv:
            prev = self.activation(self.pool(self.prev_conv[i](prev)))
        prev = prev.flatten(1)

        for i in self.prev_mlps:
            if i == 'translation':
                prev_t = (self.prev_mlps[i](prev)).unsqueeze(2)
            elif i == 'rot':
                prev_r = (self.prev_mlps[i](prev)).view(-1, 3, 3)
            else:
                prev = self.activation(self.dropout(self.prev_mlps[i](prev)))

        prev_pose = torch.cat([prev_r, prev_t], dim=2)

        for i in self.mlps:
            x = self.activation(self.mlps[i](x))

        x, _ = self.gru(x)
        x = torch.linalg.norm(x, axis=1)
        x = self.activation(self.decode_fc(x))
        r = (self.rot_fc(x)).view(-1, 3, 3)
        t = (self.transl_fc(x)).unsqueeze(2)
        x = torch.cat([r, t], dim=2)
        x -= prev_pose

        return x

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':
#     device = torch.device('cpu')
#     torch.manual_seed(55)
#     ex = torch.randn(1, 3, 256, 832, device=device)
#     posenet = PoseNet().to(device)
#     x = posenet(ex, ex)
#     print(x.size())
#     print(x)

#     model = KFNet().to(device)
#     y = model(ex, ex)

#     print(y.size())
#     print(y)
