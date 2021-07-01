import cv2 as cv
import numpy as np
import torch


def display_depth(model, transform, device, video, height=512, width=512):
    alpha = 0.2
    beta = 1 - alpha
    model.load()
    model.to(device)
    frame_tensor = torch.zeros((1, 3, height, width),
                               device=device,
                               dtype=torch.float32)

    cap = cv.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        v_frame = frame.copy()
        v_frame = cv.resize(v_frame, (834, 256))
        v_frame = np.float32(v_frame / 255)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_tensor[0] = transform(frame)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = model(frame_tensor)[0]
                pred = pred.permute(1, 2, 0)
        frame = pred.detach().cpu().numpy().astype(np.float32)

        frame = cv.resize(frame, (834, 256))

        dst = cv.addWeighted(v_frame, alpha, frame, beta, 0.0)

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
