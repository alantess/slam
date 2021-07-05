import os
import torch
import cv2 as cv
import glob
import numpy as np
import random
import torchvision
from torch.utils.data import Dataset


class KittiSeq(Dataset):
    def __init__(self,
                 root,
                 pose_root=None,
                 train=True,
                 transforms=None,
                 seed=99):
        random.seed(seed)
        np.random.seed(seed)
        self.root = root
        self.pose_root = pose_root
        self.train = train
        self.items = None
        self.transforms = transforms
        self.mode = "train.txt" if self.train else "val.txt"
        if pose_root:
            self.pose_seq = sorted(glob.glob(os.path.join(pose_root, "*.txt")))
        self.folders = [f[:-1] for f in open(root + self.mode)]
        self.items = None

        self.combine_folder()

    def combine_folder(self):
        seq_set = []

        for folder in self.folders:
            pose_file_idx = int(folder[:2])
            pose_file = self.pose_seq[pose_file_idx]
            pose = np.genfromtxt(pose_file).astype(np.float32)

            access_point = self.root + folder
            imgs = sorted(glob.glob(os.path.join(access_point, "*.jpg")))
            cam_file = os.path.join(access_point, "cam.txt")
            intrinsic = np.genfromtxt(cam_file).astype(np.float32).reshape(
                (3, 3))
            n = len(imgs)
            for i in range(0, n - 1, 2):
                sample = {
                    "intrinsic": intrinsic,
                    "tgt": imgs[i],
                    "ref": imgs[i + 1],
                    "pose": pose[i + 1].reshape(3, 4)
                }
                seq_set.append(sample)
        self.items = seq_set

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample = self.items[idx]
        tgt = cv.imread(sample['tgt']).astype(np.float32)
        ref = cv.imread(sample['ref']).astype(np.float32)
        intrinsic = np.copy(sample['intrinsic'])
        pose = sample["pose"]  # extrinsic
        if self.transforms:

            tgt = self.transforms(tgt)
            ref = self.transforms(ref)
            intrinsic = torch.from_numpy(intrinsic)
            pose = torch.from_numpy(pose)
            pose = (pose - 0.406) / 0.224
            intrinsic_inv = torch.linalg.inv(intrinsic)
        else:
            intrinsic_inv = np.linalg.inv(intrinsic)

        return tgt, ref, intrinsic, intrinsic_inv, pose


class KittiDepthSet(Dataset):
    def __init__(self, root, transforms=None, train=True):
        self.transforms = transforms
        self.mode = 'train.txt' if train else 'val.txt'
        self.folders = [root + f[:-1] for f in open(root + self.mode)]
        self.total_size = 0
        self.imgs = []
        self.depth = []
        self._crawl_folders()

    def _crawl_folders(self):
        for folder in self.folders:
            real = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            depth = sorted(glob.glob(os.path.join(folder, "*.png")))
            assert len(real) == len(depth)
            n = len(real)
            self.total_size += n
            for i in range(n):
                self.imgs.append(real[i])
                self.depth.append(depth[i])

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        alpha = 0.65
        beta = 1 - alpha
        img = cv.imread(self.imgs[idx])
        depth = cv.imread(self.depth[idx])
        foreground = cv.applyColorMap(depth, cv.COLORMAP_HOT)
        background = cv.applyColorMap(depth, cv.COLORMAP_AUTUMN)
        depth = cv.addWeighted(foreground, alpha, background, beta, 0)

        if self.transforms:
            img = self.transforms(img)
            depth = self.transforms(depth)

        return img, depth
