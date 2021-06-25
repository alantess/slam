import os
import torch
import cv2 as cv
import glob
import numpy as np
import random
from torch.utils.data import Dataset


class KittiOdometry(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 seq_len=3,
                 frame_skips=1,
                 transforms=None,
                 seed=99):
        random.seed(seed)
        np.random.seed(seed)
        self.root = root
        self.train = train
        self.frame_skips = frame_skips
        self.seq_len = seq_len
        self.items = None
        self.transforms = transforms
        self.mode = "train.txt" if self.train else "val.txt"
        self.folders = [f[:-1] for f in open(root + self.mode)]
        # with open(root + self.scenes) as f:
        #     lines = f.readlines()

        #     print(lines)

        self.combine_folder()

    def combine_folder(self):
        seq_set = []
        demi_length = (self.seq_len - 1) // 2
        shifts = list(
            range(-demi_length * self.frame_skips,
                  demi_length * self.frame_skips + 1, self.frame_skips))
        shifts.pop(demi_length)
        # Access Each folder Images and intrinsic matrix
        for folder in self.folders:
            access_point = self.root + folder
            imgs = sorted(glob.glob(os.path.join(access_point, "*.jpg")))
            cam_file = os.path.join(access_point, "cam.txt")
            intrinsic = np.genfromtxt(cam_file).astype(np.float32).reshape(
                (3, 3))
            if len(imgs) < self.seq_len:
                continue
            for i in range(demi_length * self.frame_skips,
                           len(imgs) - demi_length * self.frame_skips):
                sample = {
                    'intrinsic': intrinsic,
                    'tgt': imgs[i],
                    'ref_img': []
                }
                for j in shifts:
                    sample['ref_img'].append(imgs[i + j])
                seq_set.append(sample)
            random.shuffle(seq_set)
            self.items = seq_set

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample = self.items[idx]
        tgt = cv.imread(sample['tgt']).astype(np.float32)
        ref = cv.imread(sample['ref_img'][0]).astype(np.float32)
        intrinsic = np.copy(sample['intrinsic'])
        if self.transforms:
            tgt = self.transforms(tgt)
            ref = self.transforms(ref)
            intrinsic = torch.from_numpy(intrinsic)
            intrinsic = torch.linalg.inv(intrinsic)
        else:
            intrinsic = np.linalg.inv(intrinsic)

        return tgt, ref, intrinsic
