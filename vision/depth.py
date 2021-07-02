import cv2 as cv
from torchvision import transforms
import numpy as np
import torch


def display_depth(model, transform, device, video, height=512, width=512):
    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    alpha = 0.01
    beta = 1 - alpha
    model.load()
    model.to(device)
    frame_tensor = torch.zeros((1, 3, height, width),
                               device=device,
                               dtype=torch.float32)

    cap = cv.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        v_frame = frame
        v_frame = cv.resize(v_frame, (834, 256))
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_tensor[0] = transform(frame)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = model(frame_tensor)
                pred = invTrans(pred)
                pred = pred[0].permute(1, 2, 0).mul(255).clamp(0, 255)
        frame = pred.detach().cpu().numpy().astype(np.uint8)

        frame = cv.resize(frame, (834, 256))
        frame = cv.applyColorMap(frame, cv.COLORMAP_BONE)
        frame = cv.convertScaleAbs(frame, alpha=2.7, beta=1)

        dst = cv.hconcat([v_frame, frame])

        cv.imshow('frame', dst)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


def show_dataloader(loader):
    img, depth = next(iter(loader))
    print(depth.size())
    print(torch.unique(depth[0]))
    for i, (_, depth) in enumerate(loader):

        depth = depth[0]
        depth = depth.permute(1, 2, 0).numpy()
        cv.imshow('frame', depth)
        if cv.waitKey(1) == ord('q'):
            break
