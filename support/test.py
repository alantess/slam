import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

LOSS_FN = torch.nn.MSELoss()


def test_depth(model, data_loader):
    print("****** Testing Depth ******")
    writer = SummaryWriter()
    model.load()
    model.to(DEVICE)
    loop = tqdm(data_loader)
    for i, (img, tgt, depth, _, _, _) in enumerate(loop):
        s = img.to(DEVICE, dtype=torch.float32)
        s_ = tgt.to(DEVICE, dtype=torch.float32)
        depth = depth.to(DEVICE)
        with torch.no_grad():
            pred = model(s, s_)
            loss = LOSS_FN(pred, depth)
            writer.add_scalar('Loss/Test (DEPTH)', loss.item(), i)

    writer.close()
    print("Completed.")


def test_pose(model, data_loader):
    print("****** Testing Pose ******")
    loop = tqdm(data_loader)
    writer = SummaryWriter()
    model.load()
    model.to(DEVICE)
    for i, (_, _, depth, Rt, _, k_inv) in enumerate(loop):
        k_inv = k_inv.to(DEVICE, dtype=torch.float32)
        Rt = Rt.to(DEVICE, torch.float32)
        depth = depth.to(DEVICE)
        with torch.no_grad():
            pred = model(depth, k_inv)
            loss = LOSS_FN(pred, Rt)
            writer.add_scalar('Loss/Test (POSE)', loss.item(), i)

    writer.close()
    print("Completed.")
