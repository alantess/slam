import sys

sys.path.insert(0, "..")
from vision.ptcloud import *

# To be deleted
from dataset.data import *
from torch.utils.data import DataLoader

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                     std=[0.22803, 0.22145, 0.216989])
])
path = "/media/alan/seagate/datasets/kitti/cpp/"
dataset = KittiSet(path, preprocess)
loader = DataLoader(dataset, batch_size=1)
s, s_, depth, rt, k, k_inv = next(iter(loader))
s, s_, depth2, rt, k, k_inv = next(iter(loader))


def visualize(pt):
    for i, (_, img, d, _, _, _) in enumerate(loader):

        pt.run(d)


import cv2 as cv
if __name__ == '__main__':

    pt = PointCloud(k[0])
    pt.init()
    visualize(pt)
    rt = (rt[0] * dataset.std) + dataset.mean
    transl = rt[:, 3:]
    print(rt)
