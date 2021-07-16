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
    def forward(self, prev, cur):
        state_covar, state_mean = self.scoord(cur)  #Bx2048x1 & Bx2048x3
        cur = self.cost_volume(cur)
        prev = self.cost_volume(prev)
        encode = torch.cat([prev, cur], dim=1)
        # Prior noise covariance and prior state mean
        process_noise_covar, process_mean = self.o_flow(
            encode)  #Bx2048x3 & Bx2048x1

        mean_t, covar_t = self.kalman_filter(process_mean, process_noise_covar,
                                             state_mean, state_covar)

        pose = self.pose_estimator(mean_t)

        return pose

    def kalman_filter(self, prev_mean, prev_covariance, mean_t, covar_t):
        estimated_mean = mean_t - prev_mean  # innovation
        estimated_covar = covar_t - prev_covariance
        k_t = prev_covariance / torch.sqrt(prev_covariance + covar_t)
        mean = estimated_mean + (k_t * estimated_mean)
        covar = estimated_covar * (1 - k_t)
        return mean, covar

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


# if __name__ == '__main__':
#     device = torch.device('cuda')
#     torch.manual_seed(55)
#     ex = torch.randn(1, 3, 256, 832, device=torch.device('cuda'))
#     model = KFNet().to(device)
#     y = model(ex, ex)
#     print(y.size())
