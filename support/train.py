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

    scaler = GradScaler()
    best_score = np.inf
    model = model.to(device)

    if load_model:
        model.load()
        print('MODEL LOADED.')

    print("---- Training Depth ----")
    for epoch in range(epochs):
        loop = tqdm(train_loader)
        total_loss = 0
        val_loss = 0
        # Training Loop
        for i, (img, tgt, depth, _, _, _) in enumerate(loop):
            img = img.to(device, dtype=torch.float32)
            tgt = tgt.to(device, dtype=torch.float32)
            depth = depth.to(device)

            for p in model.parameters():
                p.grad = None
            # Forward
            with autocast():
                pred_depth = model(img, tgt)  # Bx1xWXH
                loss = loss_fn(pred_depth, depth)

            # Backwards
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        # Validation Loop
        print('Validation')
        val_loop = tqdm(val_loader)
        with torch.no_grad():
            for j, (img, tgt, depth, _, _, _) in enumerate(val_loop):
                img = img.to(device, dtype=torch.float32)
                tgt = tgt.to(device, dtype=torch.float32)
                depth = depth.to(device)

                with autocast():
                    pred_depth = model(img, tgt)

                    v_loss = loss_fn(pred_depth, depth)
                val_loss += v_loss.item()
                val_loop.set_postfix(val_loss=v_loss.item())

        # Save the model depending on performance
        if val_loss < best_score:
            best_score = val_loss
            model.save()
            # Save epoch, optimizer, and loss
            print("MODEL SAVED.")

        print(
            f"Epoch #{epoch} Loss:\n(Training): {total_loss:.5f} \t (Validation): {val_loss:.5f}  "
        )


# Pose
def train_pose(model,
               depth_model,
               train_loader,
               val_loader,
               optimizer,
               loss_fn,
               device,
               epochs,
               load_model=False):
    """
    Trains both networks simultaneously
    :param model: Input model 
    :param train_loader: Training Set  
    :param val_loader: Validation Set 
    :param optimizer: Optimizer  
    :param loss_fn: Loss function 
    :param device: GPU or CPU 
    :param epochs: Training iteration 
    :param load_model: Loads saved model 
    :return: None 
    """
    scaler = GradScaler()
    best_score = np.inf
    w1, w2 = 0.1, 0.1

    depth_model.load()
    depth_model.to(device)
    if load_model:
        model.load()
        print('MODEL LOADED.')

    model.to(device)

    print("---- Training Pose ----")
    for epoch in range(epochs):
        loop = tqdm(train_loader)
        total_loss = 0
        val_loss = 0
        # Training Loop
        for i, (img, tgt, _, Rt, _, _) in enumerate(loop):
            img = img.to(device, dtype=torch.float32)
            tgt = tgt.to(device, dtype=torch.float32)
            Rt = Rt.to(device, dtype=torch.float32)

            for p in model.parameters():
                p.grad = None

            with autocast():
                with torch.no_grad():
                    depth = depth_model(img, tgt)

                pose = model(img, tgt, depth)

                err1, err2 = compute_pose_loss(pose, Rt)
                loss = (err1 * w1) + (err2 * w2)
                loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        # Validation Loop
        print('Validation')
        val_loop = tqdm(val_loader)
        with torch.no_grad():
            for j, (img, tgt, _, Rt, _, _) in enumerate(val_loop):
                img = img.to(device, dtype=torch.float32)
                tgt = tgt.to(device, dtype=torch.float32)
                Rt = Rt.to(device, dtype=torch.float32)

                with autocast():
                    with torch.no_grad():
                        depth = depth_model(img, tgt)
                    pose = model(img, tgt, depth)
                    v_err1, v_err2 = compute_pose_loss(pose, Rt)
                    v_loss = (v_err1 * w1) + (v_err2 * w2)
                    v_loss = v_loss.mean()

                val_loss += v_loss.item()
                val_loop.set_postfix(val_loss=v_loss.item())

        # Save the model depending on performance
        if val_loss < best_score:
            best_score = val_loss
            model.save()
            # Save epoch, optimizer, and loss
            print("\nMODEL SAVED.")

        print(
            f"Epoch #{epoch} Loss:\n(Training): {total_loss:.5f} \t (Validation): {val_loss:.5f}  "
        )
