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
        # Input Convs
        self.cost_volume = nn.Conv2d(3, 8, 1, 1)
        # SCoordNet (Measurement Systems)
        self.scoord = SCoordNet()
        # OFlow (Process Systems)
        self.o_flow = OFlowNet()
        # Pose estimation
        self.pose_estimator = PoseEstimator()

        # Filtering System
    def forward(self, prev, cur, prior_mean=1, prior_noise=1):
        # State obs and measurement noise covariance
        v_t, z_t = self.scoord(cur)  #Bx2048x3 & Bx2048x1
        cur = self.cost_volume(cur)
        prev = self.cost_volume(prev)
        encode = torch.cat([prev, cur], dim=1)
        # Prior noise covariance and prior state mean
        w_t, g_t = self.o_flow(encode)  #Bx2048x3 & Bx2048x1

        prior_state_mean = g_t * prior_mean  # theta
        prior_state_covar = (g_t * prior_noise) + w_t  # R_t

        mean_t, covar_t = self.kalman_filter(prior_state_mean,
                                             prior_state_covar, z_t, v_t)

        pose = self.pose_estimator(mean_t)

        return pose, mean_t, covar_t

    def kalman_filter(self, prev_mean, r_t, z_t, v_t):
        estimated_mean = z_t - prev_mean  # innovation
        k_t = r_t / (r_t + v_t)  # Kalman Gain
        new_state = prev_mean + (k_t * estimated_mean)  # Update state mean
        new_r_t = r_t * (1 - k_t)  # Update state covariance

        # measuremeant_residuals = z_t - new_state
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
            "fc1": nn.Linear(208, 256),
            "fc2": nn.Linear(256, 256),
            "fc3": nn.Linear(256, 512),
        })
        self.decode_fc = nn.Linear(256, 128)
        self.rot_fc = nn.Linear(128, 9)
        self.transl_fc = nn.Linear(128, 3)
        self.gru = nn.GRU(512, 256, 4, batch_first=True)

    def forward(self, prev, cur):
        x = self.encoder(prev, cur)
        x = x.flatten(2)
        for i in self.mlps:
            x = self.activation(self.mlps[i](x))

        x, _ = self.gru(x)
        x = torch.linalg.norm(x, axis=1)
        x = self.activation(self.decode_fc(x))
        r = (self.rot_fc(x)).view(-1, 3, 3)
        t = (self.transl_fc(x)).unsqueeze(2)
        x = torch.cat([r, t], dim=2)

        return x


# if __name__ == '__main__':
#     device = torch.device('cpu')
#     torch.manual_seed(55)
#     ex = torch.randn(1, 3, 256, 832, device=device)
#     posenet = PoseNet().to(device)
#     x = posenet(ex, ex)
#     print(x.size())

# model = KFNet().to(device)
# y, mean, covar = model(ex, ex)
# print(y)
# print(mean)
# y, mean, covar = model(ex, ex, mean, covar)
# print(mean)

# print(y.size())
