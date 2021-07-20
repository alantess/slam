import sys
sys.path.insert(0, "..")
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from support.train import *
from vision.visuals import *
from dataset.data import *
from vision.depth import *
from vision.vision import *
from support.test import *
from networks.posenet import *
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
    parser.add_argument(
        '--mode',
        type=str,
        default='pose',
        help='Trains Models or Test the models. Args: [pose,depth,test]')
    parser.add_argument(
        '--video',
        type=str,
        default=
        "https://github.com/alantess/vigilantV2/releases/download/v0.1-beta/driving.mp4",
        help='Batch size of input')

    args = parser.parse_args()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    SEED = 97
    torch.manual_seed(SEED)
    BATCH_SIZE = args.batch
    PIN_MEM = True
    NUM_WORKERS = 4
    EPOCHS = 20

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    if args.mode == 'pose':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args.img_height, args.img_width)),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])
        ])
    else:
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

    depth_model = DepthNet(model_name='depthnet152.pt')

    print('=> Setting adam solver')
    depth_optim = torch.optim.Adam(depth_model.parameters(), lr=1e-4)
    pose_optim = torch.optim.Adam(pose_model.parameters(), lr=1e-6)

    loss_fn = torch.nn.SmoothL1Loss()
    print('=> Gatheing Datset')

    trainset = KittiSet(args.kitti_dir, transforms=preprocess)
    valset = KittiSet(args.kitti_dir, transforms=preprocess, train=False)

    train_loader = DataLoader(trainset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEM)
    val_loader = DataLoader(valset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEM)

    if args.mode == 'test':
        test_pose(pose_model, val_loader)
        test_depth(depth_model, val_loader)
        display_depth(model, preprocess, device, args.video, args.img_height,
                      args.img_width)

    elif args.mode == 'train':
        train_depth(depth_model, train_loader, val_loader, depth_optim,
                    loss_fn, device, EPOCHS)
    else:
        print("Please choose an available mode: [test, pose, depth]")