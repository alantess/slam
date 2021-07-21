import sys

sys.path.insert(0, "..")
from vision.ptcloud import *
from networks.depthnet import DepthNet
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


def visualize(pt, camera, model=None):
    print("Running PointCloud...")
    if model:
        model.to(device)

    t = 0
    for i, (img, tgt, depth, _, k, _) in enumerate(loader):
        k = k.to(device, dtype=torch.float32)
        depth = depth.to(device, dtype=torch.float32)

        if model:
            img = img.to(device, dtype=torch.float32)
            tgt = tgt.to(device, dtype=torch.float32)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    pred = model(img, tgt)
                depth = pred.detach().to(dtype=torch.float32).cpu()
        camera.K = k[0].float()
        xyz = camera.pixel_to_cam(depth)
        xyz = xyz.cpu().numpy()

        pt.run(xyz)
        t += 1
        if t > 10:
            break


def main():
    # model = DepthNet()
    # model.load()
    proj = CameraProjector()
    pt = PointCloud(k[0])
    pt.init()
    visualize(pt, proj, model=None)


if __name__ == '__main__':
    main()
