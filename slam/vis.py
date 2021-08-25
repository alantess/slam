import sys

sys.path.insert(0, "..")
import cv2 as cv
from vision.ptcloud import *
from networks.slamnet import SLAMNet
from dataset.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from support.compute import CameraProjector

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                     std=[0.22803, 0.22145, 0.216989])
])
inv_preprocess = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])
path = "/media/alan/seagate/datasets/kitti/cpp/"
dataset = KittiSet(path, preprocess)
loader = DataLoader(dataset, batch_size=1)
s, s_, depth, rt, k, k_inv = next(iter(loader))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

import matplotlib.pyplot as plt

def visualize(pt, camera, model=None):
    print("Running PointCloud...")
    if model:
        model.to(device)

    for i, (img, tgt, depth, _, k, _) in enumerate(loader):
        k = k.to(device, dtype=torch.float32)
        img = img.to(device, dtype=torch.float32)

        if model:
            img = img.to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    pred = model(img,k)
                    depth = pred.detach()
                # depth = pred.detach().to(dtype=torch.float32)
        # camera.K = k[0].to(dtype=torch.float64)
        # xyz = camera.pixel_to_cam(depth)
        

        xyz = depth.cpu()

        pt.run(xyz)

def run_model(model, img, k):
    pred = model(img, k).detach()



def main():
    model = SLAMNet()
    model.load()
    proj = CameraProjector()
    pt = PointCloud(k[0])
    pt.init()
    visualize(pt, proj, model)


if __name__ == '__main__':
    main()
