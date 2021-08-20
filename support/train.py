import torch
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from .compute import *


# Depth
def train_depth(model,
                train_loader,
                val_loader,
                optimizer,
                loss_fn,
                device,
                epochs,
                load_model=False):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-----MODEL PARAMS-----\nPOSE PARAMS: {params/1e6:.1f}M")

    scaler = GradScaler()
    best_score = np.inf
    model = model.to(device)

    cam = CameraProjector(device,loss_fn)

    if load_model:
        model.load()
        print('MODEL LOADED.')

    print("---- Training Depth ----")
    for epoch in range(epochs):
        loop = tqdm(train_loader)
        total_loss = 0
        val_loss = 0
        # Training Loop
        for i, (img, tgt, depth, _, K, _) in enumerate(loop):
            img = img.to(device, dtype=torch.float32)
            K = K.to(device, dtype=torch.float32)
            depth = depth.to(device, dtype=torch.float32)
            cam.K = K
            for p in model.parameters():
                p.grad = None
            # Forward
            # with autocast():
            pred= model(img, K)  # Bx1xWXH
            loss = cam.compute_loss(pred, depth)

            # loss = loss_fn(pred_depth, depth)
            loss.backward()
            optimizer.step()

            # Backwards
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        # Validation Loop
        print('Validation')
        val_loop = tqdm(val_loader)
        with torch.no_grad():
            for j, (img, tgt, depth, _, K, _) in enumerate(val_loop):
                img = img.to(device, dtype=torch.float32)
                depth = depth.to(device, dtype=torch.float32)
                K = K.to(device, dtype=torch.float32)
                cam.K = K
                # with autocast():
                out = model(img, K)
                v_loss = cam.compute_loss(out, depth)
                val_loss += v_loss.item()
                val_loop.set_postfix(val_loss=v_loss.item())

        # Save the model depending on performance
        if val_loss < best_score:
            best_score = val_loss
            model.save()
            # Save epoch, optimizer, and loss
            print("--- MODEL SAVED ---")

        print(
            f"\nEpoch #{epoch} Loss:\n(Training): {total_loss:.5f} \t (Validation): {val_loss:.5f}  "
        )
