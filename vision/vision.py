import cv2 as cv
import numpy as np
import torch
import gc
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt


def visualize_depth(model,
                    video,
                    transforms,
                    device,
                    height,
                    width,
                    inv_depth=False,
                    save_video=False):

    gc.collect()
    torch.cuda.empty_cache()
    model.load()
    model.to(device)
    s = torch.empty((1, 3, height, width), device=device)
    s_ = torch.empty((1, 3, height, width), device=device)

    print("Opening video...")
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        print("Cannot open video")
        exit()

    ret, prev_frame = cap.read()
    while ret:
        ret, cur_frame = cap.read()

        # Transform each frame into Tensor
        s[0] = transforms(prev_frame)
        s_[0] = transforms(cur_frame)
        # Predict the depth
        with torch.no_grad():
            with autocast():
                depth = model(s, s_)[0]
                if inv_depth:
                    depth = 1 / depth

                depth = depth.squeeze(0).mul(
                    255).detach().cpu().numpy().astype(np.uint8)

        depth = cv.cvtColor(depth, cv.COLOR_GRAY2RGB)
        depth = cv.applyColorMap(depth, cv.COLORMAP_BONE)

        depth = cv.convertScaleAbs(depth, alpha=1.6, beta=2)

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv.imshow('frame', depth)
        if cv.waitKey(1) == ord('q'):
            break

        prev_frame = cur_frame

    cap.release()
    cv.destroyAllWindows()
    print('COMPLETED.')


def show_depth_example(model, device, loader):
    s, s_, _, _, _, _ = next(iter(loader))
    model.load()
    model.to(device)
    s = s.to(device)
    s_ = s_.to(device)
    with torch.no_grad():
        with autocast():
            y = model(s, s_)

    y = y[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    plt.imshow(y)
    plt.show()
