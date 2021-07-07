import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from support.train import *
from vision.visuals import *
from dataset.data import *
from vision.depth import *
from networks.posenet import PoseNet
from networks.depthnet import DepthNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLAM')
    parser.add_argument('--kitti-dir',
                        type=str,
                        default="/media/alan/seagate/datasets/kitti/cpp/",
                        help='Kitti VO dataset')

    parser.add_argument("--img-height",
                        default=256,
                        type=int,
                        help="Image height")
    parser.add_argument("--img-width",
                        default=832,
                        type=int,
                        help="Image width")
    parser.add_argument('--batch',
                        type=int,
                        default=2,
                        help='Batch size of input')
    parser.add_argument('--test',
                        type=bool,
                        default=False,
                        help='Trains Models')
    parser.add_argument(
        '--video',
        type=str,
        default=
        "https://github.com/alantess/vigilantV2/releases/download/v0.1-beta/driving.mp4",
        help='Batch size of input')

    args = parser.parse_args()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    BATCH_SIZE = args.batch
    PIN_MEM = True
    NUM_WORKERS = 4
    EPOCHS = 20

    torch.backends.cudnn.benchmark = True

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_height, args.img_width)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    inv_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])

    depth_model = DepthNet()
    pose_model = PoseNet(n_layers=4)
    print('=> Setting adam solver')
    optim_params = [{
        'params': depth_model.parameters(),
        'lr': 1e-4
    }, {
        'params': pose_model.parameters(),
        'lr': 1e-4
    }]
    optimizer = torch.optim.Adam(optim_params)

    loss_fn = torch.nn.SmoothL1Loss()
    print('=> Gatheing Datset')

    # Dataset
    trainset = KittiSet(args.kitti_dir, preprocess)
    valset = KittiSet(args.kitti_dir, preprocess, False)

    train_loader = DataLoader(trainset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEM)
    val_loader = DataLoader(valset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEM)

if args.test:
    display_depth(model, preprocess, device, args.video, args.img_height,
                  args.img_width)

else:
    train(pose_model, depth_model, train_loader, val_loader, optimizer,
          loss_fn, device, EPOCHS)
