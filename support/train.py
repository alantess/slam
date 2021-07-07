import torch
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from .compute import *


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

    print("---- Starting ----")
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


def train_slom(pose_model,
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
    pose_model = pose_model.to(device)
    depth_model = depth_model.to(device)
    w1, w2, w3 = 0.2, 0.1, 0.1,

    if load_model:
        pose_model.load()
        depth_model.load()
        print('MODEL LOADED.')

    print("---- Starting ----")
    for epoch in range(epochs):
        loop = tqdm(train_loader)
        total_loss = 0
        val_loss = 0
        # Training Loop
        for i, (img, tgt, depth, Rt, intrinsic, _) in enumerate(loop):
            img = img.to(device, dtype=torch.float32)
            tgt = tgt.to(device, dtype=torch.float32)
            intrinsic = intrinsic.to(device, dtype=torch.float16)
            depth = depth.to(device)
            Rt = Rt.to(device, dtype=torch.float32)

            for p in pose_model.parameters():
                p.grad = None

            for p in depth_model.parameters():
                p.grad = None

            with autocast():
                pose = pose_model(img, tgt)
                pred_depth = depth_model(img, tgt)

                depth_loss1 = loss_fn(pred_depth, depth)
                Rt_loss1 = compute_ate(pose, Rt)
                Rt_loss2 = compute_translation(pose, Rt)
                loss = w1 * depth_loss1 + w2 * Rt_loss1 + w3 * Rt_loss2
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        # Validation Loop
        print('Validation')
        val_loop = tqdm(val_loader)
        with torch.no_grad():
            for j, (img, tgt, depth, Rt, intrinsic, _) in enumerate(val_loop):
                img = img.to(device, dtype=torch.float32)
                tgt = tgt.to(device, dtype=torch.float32)
                intrinsic = intrinsic.to(device, dtype=torch.float16)
                depth = depth.to(device)
                Rt = Rt.to(device, dtype=torch.float32)

                with autocast():
                    pose = pose_model(img, tgt)
                    pred_depth = depth_model(img, tgt)

                    depth_loss1 = loss_fn(pred_depth, depth)
                    Rt_loss1 = compute_ate(pose, Rt)
                    Rt_loss2 = compute_translation(pose, Rt)
                    v_loss = w1 * depth_loss1 + w2 * Rt_loss1 + w3 * Rt_loss2
                val_loss += v_loss.item()
                val_loop.set_postfix(val_loss=v_loss.item())

        # Save the model depending on performance
        if val_loss < best_score:
            best_score = val_loss
            pose_model.save()
            depth_model.save()
            # Save epoch, optimizer, and loss
            torch.save(
                {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, "model_checkpoints/gen_data")
            print("MODEL SAVED.")

        print(
            f"Epoch #{epoch} Loss:\n(Training): {total_loss:.5f} \t (Validation): {val_loss:.5f}  "
        )
