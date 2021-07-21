import cv2 as cv
from torchvision import transforms
import numpy as np
import torch


def display_depth(model,
                  transform,
                  device,
                  video,
                  height=512,
                  width=512,
                  colormap='bone',
                  disparity=False,
                  save_video=False):
    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])

    if save_video:
        video_size = (1668, 256)
        result = cv.VideoWriter('etc/depth.avi',
                                cv.VideoWriter_fourcc(*'MJPG'), 60, video_size)
    model.load()
    model.to(device)
    prev_frame = torch.zeros((1, 3, height, width),
                             device=device,
                             dtype=torch.float32)

    frame_tensor = torch.zeros((1, 3, height, width),
                               device=device,
                               dtype=torch.float32)

    cap = cv.VideoCapture(video)

    ret, frame = cap.read()
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
                if disparity:
                    pred = 1 / pred
        frame = pred.detach().cpu().numpy().astype(np.uint8)

        frame = cv.resize(frame, (834, 256))
        if colormap == 'bone':
            frame = cv.applyColorMap(frame, cv.COLORMAP_BONE)
        if colormap == 'hot':
            frame = cv.applyColorMap(frame, cv.COLORMAP_HOT)

        frame = cv.convertScaleAbs(frame, alpha=2.9, beta=1)

        dst = cv.hconcat([v_frame, frame])
        if save_video:
            result.write(dst)

        cv.imshow('frame', dst)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    if save_video:
        result.release()
        print("Video saved. \n Check 'etc' folder ")
    cv.destroyAllWindows()
