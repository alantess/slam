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
