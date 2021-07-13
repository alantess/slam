import os
import torch
import cv2 as cv
import glob
import numpy as np
import torchvision
from torch.utils.data import Dataset


class KittiSet(Dataset):
    def __init__(self,
                 root,
                 transforms=None,
                 train=True,
                 frame_skip=True,
                 make_sequential=True,
                 video_frames=16):
        self.transforms = transforms
        self.make_sequential = make_sequential
        self.video_frames = video_frames
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

            if not self.make_sequential:
                for i in range(0, n - 1, inc):
                    sample = {
                        "frame": imgs[i],
                        "next_frame": imgs[i + 1],
                        "depth": depth[i + 1],
                        "poses": poses[i + 1].reshape(3, 4),
                        "intrinsic": k,
                    }
                    seq_set.append(sample)
            else:
                for i in range(0, n - self.video_frames, self.video_frames):
                    sample = {
                        'frames': imgs[i:i + self.video_frames],
                        'poses': poses[i + self.video_frames].reshape(3, 4),
                        "intrinsic": k
                    }
                    seq_set.append(sample)

        self.samples = seq_set

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if not self.make_sequential:
            s = cv.imread(sample["frame"])  #HxWxC
            s_ = cv.imread(sample["next_frame"])
            depth = cv.imread(sample["depth"])
        else:
            frames = np.empty((self.video_frames, 256, 832, 3))
            for i in range(len(frames)):
                frames[i] = cv.imread(sample["frames"][i])

        Rt = sample["poses"]
        k = sample["intrinsic"]
        k_inv = np.linalg.inv(k)

        if self.transforms:
            if not self.make_sequential:
                grayscale = torchvision.transforms.Grayscale()
                s = self.transforms(s)
                s_ = self.transforms(s_)
                depth = self.transforms(depth)
                depth = grayscale(depth)
            else:
                frame_tensor = torch.empty((self.video_frames, 3, 256, 832))
                for i in range(len(frame_tensor)):
                    frame_tensor[i] = self.transforms(frames[i])

                frames = frame_tensor.permute(1, 0, 2, 3)
            Rt = torch.from_numpy(Rt)
            k = torch.from_numpy(k)
            k_inv = torch.from_numpy(k_inv)

        if not self.make_sequential:
            return s, s_, depth, Rt, k, k_inv
        else:
            return frames, Rt, k


# if __name__ == '__main__':
#     from torch.utils.data import DataLoader

#     preprocess = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
#                                          std=[0.22803, 0.22145, 0.216989])
#     ])
#     path = "/media/alan/seagate/datasets/kitti/cpp/"
#     dataset = KittiSet(path, preprocess)
#     loader = DataLoader(dataset, batch_size=16)
#     u, x, y = next(iter(loader))
