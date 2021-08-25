import sys
import os

sys.path.insert(0, "..")
import torch
import time
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from support.train import *
from dataset.data import *
from vision.vision import *
from support.test import *
from networks.slamnet import SLAMNet



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["OMP_SCHEDULE"]="STATIC"
    os.environ["OMP_PROC_BIND"]="CLOSE"
    os.environ["GOMP_CPU_AFFINITY"] = "0-16"
    os.environ["KMP_BLOCKTIME"] = "1"

    start_time = time.time();
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
        default='train',
        help='Trains Models or Test the models.')
    parser.add_argument(
        '--video',
        type=str,
        default=
        "https://github.com/alantess/vigilantV2/releases/download/v0.1-beta/driving.mp4",
        help='Batch size of input')

    args = parser.parse_args()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    g = torch.Generator()
    g.manual_seed(0)
    SEED = 97
    torch.manual_seed(SEED)
    BATCH_SIZE = args.batch
    PIN_MEM = True
    NUM_WORKERS = 4
    EPOCHS = 20

    torch.cuda.empty_cache()
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

    print('=> Setting up network')

    slam_model = SLAMNet()

    slam_optim = torch.optim.Adam(slam_model.parameters(), lr=3e-6)

    loss_fn = torch.nn.SmoothL1Loss()
    print('=> Gatheing Datset')

    trainset = KittiSet(args.kitti_dir, transforms=preprocess)
    valset = KittiSet(args.kitti_dir, transforms=preprocess, train=False)

    train_loader = DataLoader(trainset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True,
                              worker_init_fn=seed_worker,
                              pin_memory=PIN_MEM
                              )
    val_loader = DataLoader(valset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker,
                            pin_memory=PIN_MEM)

    if args.mode == 'test':
        show_depth_example(slam_model, device, val_loader)
        visualize_depth(slam_model, args.video, preprocess, device,
                        args.img_height, args.img_width)
        # test_depth(slam_model, val_loader)
    elif args.mode == 'train':
        train_depth(slam_model, train_loader, val_loader, slam_optim,
                    loss_fn, device, EPOCHS)
    else:
        print("Please choose an available mode: [train, test]")

    end_time = time.time()
    print(f"Execution Time {end_time-start_time:.3f}")
