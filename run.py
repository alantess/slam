import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from support.train import train
from networks.depth import DisparityNet
from dataset.data import NYUDepth

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
                        default=834,
                        type=int,
                        help="Image height")
    parser.add_argument("--img-width",
                        default=256,
                        type=int,
                        help="Image width")
    parser.add_argument('--batch',
                        type=int,
                        default=4,
                        help='Batch size of input')

    args = parser.parse_args()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    BATCH_SIZE = args.batch
    PIN_MEM = True
    NUM_WORKERS = 4
    EPOCHS = 200

    torch.backends.cudnn.benchmark = True

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_height, args.img_width)),
    ])

    model = DisparityNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.MSELoss()

    trainset = NYUDepth(args.disparity_dir, preprocess)
    train_loader = DataLoader(trainset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEM)
    valset = NYUDepth(args.disparity_dir, preprocess, False)
    val_loader = DataLoader(valset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEM)

    train(model, train_loader, val_loader, optimizer, loss_fn, device, EPOCHS)
