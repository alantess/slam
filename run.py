import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from support.train import train
from networks.depth import DisparityNet
from dataset.data import *
from vision.depth import display_depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLAM')
    parser.add_argument('--disparity-dir',
                        type=str,
                        default="/media/alan/seagate/dataset/depth_perception",
                        help="NYU Depth dataset")
    parser.add_argument(
        '--kitty-dir',
        type=str,
        default="/media/alan/seagate/datasets/kitti/vo/kitti_vo_256",
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
                        default=4,
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
    EPOCHS = 50

    torch.backends.cudnn.benchmark = True

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((args.img_height, args.img_width)),
    ])

    model = DisparityNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    trainset = DepthDataset(args.disparity_dir, preprocess)
    train_loader = DataLoader(trainset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEM)
    valset = DepthDataset(args.disparity_dir, preprocess, False)
    val_loader = DataLoader(valset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEM)

    train(model, train_loader, val_loader, optimizer, loss_fn, device, EPOCHS)
    display_depth(model, preprocess, device, args.video)
