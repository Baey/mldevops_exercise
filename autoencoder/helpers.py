import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable

def train(model: nn.Module, train_dl: DataLoader, test_dl: DataLoader, loss_fn: Callable, optimizer: Callable, epochs: int, run = None) -> dict:
    """
    Trains the given model using the provided training and testing data loaders, loss function, and optimizer.
    Args:
        model (nn.Module): The neural network model to be trained.
        train_dl (DataLoader): DataLoader for the training dataset.
        test_dl (DataLoader): DataLoader for the testing dataset.
        loss_fn (Callable): Loss function to be used for training.
        optimizer (Callable): Optimizer to be used for training.
        epochs (int): Number of epochs to train the model.
        run (optional): An optional object for logging metrics (e.g., a run object from a logging library).
    Returns:
        dict: A dictionary containing the training and testing loss history with keys 'train_loss' and 'test_loss'.
    """
    
    test_loss_history = []
    train_loss_history = []
    for i in range(epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        train_loss = _train_step(train_dl, model, loss_fn, optimizer)
        test_loss = _evaluate(test_dl, model, loss_fn)
        if run is not None:
            run.log({"train_loss": train_loss, "test_loss": test_loss})
        test_loss_history.append(test_loss)
        train_loss_history.append(train_loss)
    print("Done!")
    return {'train_loss': train_loss_history, 'test_loss': test_loss_history}

def _train_step(dataloader: DataLoader, model: nn.Module, loss_fn: Callable, optimizer: Callable) -> float:
    """
    Perform a single training step for an autoencoder model.
    Args:
        dataloader (DataLoader): DataLoader providing the training data.
        model (nn.Module): The autoencoder model to be trained.
        loss_fn (Callable): The loss function to compute the reconstruction loss.
        optimizer (Callable): The optimizer used to update the model parameters.
    Returns:
        float: The average loss over the entire dataset for this training step.
    """

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
    """
    Evaluate the performance of a model on a given dataloader using a specified loss function.
    Args:
        dataloader (DataLoader): The DataLoader providing the data to evaluate.
        model (nn.Module): The neural network model to evaluate.
        loss_fn (Callable): The loss function used to compute the loss.
    Returns:
        float: The average loss over all batches in the dataloader.
    """

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