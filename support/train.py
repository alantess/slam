import torch
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler


def train(model,
          train_loader,
          val_loader,
          optimizer,
          loss_fn,
          device,
          epochs,
          load_model=False):
    """
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
        for i, (img, tgt) in enumerate(loop):
            img = img.to(device, dtype=torch.float32)
            tgt = tgt.to(device, dtype=torch.float32)

            for p in model.parameters():
                p.grad = None

            with autocast():
                pred = model(img)
                loss = loss_fn(pred, tgt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        # Validation Loop
        print('Validation')
        val_loop = tqdm(val_loader)
        with torch.no_grad():
            for j, (img, tgt) in enumerate(val_loop):
                img = img.to(device, dtype=torch.float32)
                tgt = tgt.to(device, dtype=torch.float32)

                with autocast():
                    v_pred = model(img)
                    v_loss = loss_fn(v_pred, tgt)

                val_loss += v_loss.item()
                val_loop.set_postfix(val_loss=v_loss.item())

        # Save the model depending on performance
        if val_loss < best_score:
            best_score = val_loss
            model.save()
            print("MODEL SAVED.")

        print(
            f"Epoch #{epoch} Loss:\n(Training): {total_loss:.5f} \t (Validation): {val_loss:.5f}  "
        )