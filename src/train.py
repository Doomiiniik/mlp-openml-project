import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, optimizer, device, grad_clip=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)




def validate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item()

            predicted = preds.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy





from torch.utils.tensorboard import SummaryWriter
import torch
import os

def train_model(model, train_loader, val_loader, optimizer, device,
                epochs, grad_clip=None, log_dir="runs", save_path="best_model.pt"):
    
    writer = SummaryWriter(log_dir)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        val_loss, val_acc = validate(model, val_loader, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    writer.close()
    return best_val_loss







def load_best_model(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


'''

def train_final_model(model, train_loader, val_loader, optimizer, device,
                      epochs, grad_clip=None, save_path="models/best_model.pt",
                      log_dir="runs/final"):
    
    writer = SummaryWriter(log_dir)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        val_loss, val_acc = validate(model, val_loader, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    writer.close()
    return best_val_loss
'''

def train_final_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    grad_clip=None,
    save_path="models/best_model.pt",
    log_dir="runs/final"
):
    from torch.utils.tensorboard import SummaryWriter
    import torch.nn as nn
    import torch

    writer = SummaryWriter(log_dir)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # -------------------------
        # TRAINING
        # -------------------------
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # log train loss
        writer.add_scalar("Loss/train", train_loss, epoch)

        # -------------------------
        # OPTIONAL VALIDATION
        # -------------------------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    logits = model(X)
                    loss = criterion(logits, y)
                    val_loss += loss.item()

                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            val_loss /= len(val_loader)
            val_acc = correct / total

            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            # save best model based on validation
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)

            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        else:
            # -------------------------
            # FINAL TRAINING (NO VAL)
            # -------------------------
            torch.save(model.state_dict(), save_path)

            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} (no validation)")

    writer.close()
