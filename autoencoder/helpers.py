import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable

def train(model: nn.Module, train_dl: DataLoader, test_dl: DataLoader, loss_fn: Callable, optimizer: Callable, epochs: int, wandb_logging: bool = True) -> dict:
    test_loss_history = []
    train_loss_history = []
    for i in range(epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        train_loss = _train_step(train_dl, model, loss_fn, optimizer)
        test_loss = _evaluate(test_dl, model, loss_fn)
        if wandb_logging:
            wandb.log({"train_loss": train_loss, "test_loss": test_loss})
        test_loss_history.append(test_loss)
        train_loss_history.append(train_loss)
    print("Done!")
    return {'train_loss': train_loss_history, 'test_loss': test_loss_history}

def _train_step(dataloader: DataLoader, model: nn.Module, loss_fn: Callable, optimizer: Callable) -> float:
    device = next(model.parameters()).device
    size = len(dataloader.dataset)
    total_loss = 0

    model.train()

    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device)
        
        reconstructed = model(X)
        loss = loss_fn(reconstructed, X)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss, current = loss.item(), batch*len(X)
        total_loss += loss
        if batch % 100 == 0:
            print(f'loss:{loss:>7f} [{current:>5d}/{size:>5d}]')

    return total_loss/len(dataloader)

def _evaluate(dataloader: DataLoader, model: nn.Module, loss_fn: Callable) -> float:
    num_batches = len(dataloader)
    test_loss = 0
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, X).item()
    test_loss /= num_batches

    print(f'Test Error: \n Avg loss: {test_loss:>8f} \n')
    return test_loss