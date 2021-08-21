import os
import torch
import cv2 as cv
import glob
import numpy as np
import torchvision
from support.compute import CameraProjector
from torch.utils.data import Dataset


class KittiSet(Dataset):
    def __init__(self,
                 root,
                 transforms=None,
                 train=True,
                 frame_skip=True,
                 video_frames=16):
        self.transforms = transforms
        self.video_frames = video_frames
        self.mode = 'train.txt' if train else 'val.txt'
        self.folders = [root + f[:-1] for f in open(root + self.mode)]
        self.total_size = 0
        self.frame_skip = frame_skip
        self.mean = 0.0
        self.std = 0.229
        self.samples = None
        self._crawl_folders()
        self.cam = CameraProjector()

    def _crawl_folders(self):
        seq_set = []
        rot_mean, rot_std, t_mean, t_std = 0, 0, 0, 0
        for folder in self.folders:
            imgs = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            depth = sorted(glob.glob(os.path.join(folder, "*.png")))
            poses_file = os.path.join(folder, "poses.txt")
            cam_file = os.path.join(folder, "cam.txt")
            k = np.genfromtxt(cam_file).reshape((3, 3))
            poses = np.genfromtxt(poses_file)

            assert len(imgs) == len(depth)
            n = len(poses)

            if self.frame_skip:
                inc = 3
            else:
                inc = 1

            for i in range(0, n - 1, inc):
                sample = {
                    "frame": imgs[i],
                    "next_frame": imgs[i + 1],
                    "depth": depth[i + 1],
                    "poses": poses[i + 1].reshape(3, 4),
                    "intrinsic": k,
                }
                x = sample['poses']
                rot_mean += (x[:, :3] - x[:, :3].min())
                rot_std += (x[:, :3].max() - x[:, :3].min())
                t_mean += (x[:, 3:] - x[:, 3:].min())
                t_std += (x[:, 3:].max() - x[:, 3:].min())

                seq_set.append(sample)

        self.samples = seq_set
        # Normalization Settings
        t_mean[1, :] *= 5
        self.mean = np.concatenate([rot_mean, t_mean], axis=1)
        self.std = np.mean(rot_std - t_std)
        self.mean /= int(len(self.samples))
        self.std /= (int(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        s = cv.imread(sample["frame"])  #HxWxC
        s_ = cv.imread(sample["next_frame"])
        depth = cv.imread(sample["depth"]).astype(np.float32)
        Rt = sample["poses"]
        k = sample["intrinsic"]
        k_inv = np.linalg.inv(k)

        if self.transforms:
            grayscale = torchvision.transforms.Grayscale()
            s = self.transforms(s)
            s_ = self.transforms(s_)
            depth = self.transforms(depth)
            depth = grayscale(depth)
            Rt = torch.from_numpy(Rt)
            Rt = (Rt - self.mean) / self.std
            k = torch.from_numpy(k)

        return s, s_, depth, Rt, k,k_inv


# if __name__ == '__main__':
#     from torch.utils.data import DataLoader

#     preprocess = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
#                                          std=[0.22803, 0.22145, 0.216989])
#     ])
#     path = "/media/alan/seagate/datasets/kitti/cpp/"
#     dataset = KittiSet(path, preprocess)
#     loader = DataLoader(dataset, batch_size=1)
#     s, s_, d, rt, k, k_inv = next(iter(loader))
# print(rt)
