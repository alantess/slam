import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from support.train import train
from networks.depth import DisparityNet
from networks.ures import URes152
from dataset.data import *
from vision.depth import display_depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLAM')
    parser.add_argument('--disparity-dir',
                        type=str,
                        default="/media/alan/seagate/dataset/depth_perception",
                        help="NYU Depth dataset")
    parser.add_argument(
        '--kitti_vo_dir',
        type=str,
        default="/media/alan/seagate/datasets/kitti/vo/kitti_vo_256",
        help='Kitti VO dataset')
    parser.add_argument(
        '--kitti-depth-dir',
        type=str,
        default="/media/alan/seagate/datasets/kitti/256/kitti_256/",
        help='Kitti VO dataset')

    parser.add_argument('--nyu-dir',
                        type=str,
                        default="/media/alan/seagate/Downloads/nyudepth",
                        help='Kitti VO dataset')

    parser.add_argument("--img-height",
                        default=512,
                        type=int,
                        help="Image height")
    parser.add_argument("--img-width",
                        default=512,
                        type=int,
                        help="Image width")
    parser.add_argument('--batch',
                        type=int,
                        default=8,
                        help='Batch size of input')
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
    EPOCHS = 30

    torch.backends.cudnn.benchmark = True

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_height, args.img_width)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model = DisparityNet(n_out_channels=3, model_name='depth.pt')
    # model = URes152(n_channels=3, model_name='ureskitti.pt')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.L1Loss()

    trainset = KittiDepthSet(args.kitti_depth_dir, preprocess)
    valset = KittiDepthSet(args.kitti_depth_dir, preprocess, False)

    # trainset = DepthDataset(args.disparity_dir, preprocess)
    # valset = DepthDataset(args.disparity_dir, preprocess, False)

    # NYU V2 Depth
    # trainset = NYUDepth(args.nyu_dir, preprocess)
    # valset = NYUDepth(args.nyu_dir, preprocess, False)

    train_loader = DataLoader(trainset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEM)
    val_loader = DataLoader(valset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEM)

    # train(model, train_loader, val_loader, optimizer, loss_fn, device, EPOCHS)
    display_depth(model, preprocess, device, args.video, args.img_height,
                  args.img_width)
