import cv2 as cv
import numpy as np
import torch


def display_depth(model, transform, device, video, height=512, width=512):
    model.load()
    model.to(device)
    frame_tensor = torch.zeros((1, 3, height, width),
                               device=device,
                               dtype=torch.float32)

    cap = cv.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
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

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
