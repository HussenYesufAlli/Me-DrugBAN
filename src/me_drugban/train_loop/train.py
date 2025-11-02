import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        graphs = batch["graph"].to(device)
        proteins = batch["protein"].to(device)
        labels = batch["label"].float().to(device)

        outputs = model(graphs, proteins).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        all_preds.append(outputs.detach().cpu())
        all_labels.append(labels.cpu())

    avg_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = ((all_preds > 0.5).float() == all_labels).float().mean().item()
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            graphs = batch["graph"].to(device)
            proteins = batch["protein"].to(device)
            labels = batch["label"].float().to(device)

            outputs = model(graphs, proteins).squeeze(-1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.cpu())

    avg_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = ((all_preds > 0.5).float() == all_labels).float().mean().item()
    return avg_loss, acc

def train(model, train_loader, val_loader, epochs, optimizer, criterion, device, scheduler=None, save_path=None):
    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step(val_loss)
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  Best model saved to {save_path}")

    return history