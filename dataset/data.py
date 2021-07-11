import os
import torch
import cv2 as cv
import glob
import numpy as np
import random
import torchvision
from torch.utils.data import Dataset


class KittiSet(Dataset):
    def __init__(self,
                 root,
                 transforms=None,
                 train=True,
                 frame_skip=True,
                 mean=0.485,
                 std=0.229):
        self.transforms = transforms
        self.mean = mean
        self.std = std
        self.mode = 'train.txt' if train else 'val.txt'
        self.folders = [root + f[:-1] for f in open(root + self.mode)]
        self.total_size = 0
        self.frame_skip = frame_skip
        self.samples = None
        self._crawl_folders()

    def _crawl_folders(self):
        seq_set = []
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
                seq_set.append(sample)

        self.samples = seq_set

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        s = cv.imread(sample["frame"])
        s_ = cv.imread(sample["next_frame"])
        depth = cv.imread(sample["depth"])
        Rt = sample["poses"]
        k = sample["intrinsic"]
        k_inv = np.linalg.inv(k)

        if self.transforms:
            grayscale = torchvision.transforms.Grayscale()
            s = self.transforms(s)
            s_ = self.transforms(s_)
            depth = self.transforms(depth)
            depth = grayscale(depth)
            Rt = (Rt - self.mean) / self.std
            # k = (k - self.mean) / self.std
            Rt = torch.from_numpy(Rt)
            k = torch.from_numpy(k)
            k_inv = torch.from_numpy(k_inv)

        return s, s_, depth, Rt, k, k_inv


# if __name__ == '__main__':
# path = "/media/alan/seagate/datasets/kitti/cpp/"
# dataset = KittiSet(path)
