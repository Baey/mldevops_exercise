import os
import torch
import optuna
import datetime
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from optuna.trial import TrialState

from autoencoder import Autoencoder as AE
from autoencoder import train


def objective(trial: optuna.Trial):
    embedding_dim = 100
    batch_size = trial.suggest_int('batch_size', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    epochs = trial.suggest_int('epochs', 10, 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor(),)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor(),)

    train_dl= DataLoader(train_data, batch_size=batch_size)
    test_dl = DataLoader(test_data, batch_size=batch_size)

    model = AE(embedding_dim=embedding_dim).to(device)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = learning_rate,
        weight_decay = weight_decay
        )
    
    history = train(model, train_dl, test_dl, loss_fn, optimizer, epochs, trail=trial)

    return history['test_loss'][-1]


def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    main()