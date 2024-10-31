import torch
import wandb
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from autoencoder import Autoencoder as AE
from autoencoder import train


def main(embedding_dim: int, batch_size: int, learning_rate: float, weight_decay: float, epochs: int, device: str):
    wandb.init(
        # set the wandb project where this run will be logged
        project="MNIST-Autoencoder",
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "CNN",
        "dataset": "MNIST",
        "epochs": epochs,
        }
    )
    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor(),)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor(),)

    train_dl= DataLoader(train_data, batch_size=batch_size)
    test_dl = DataLoader(test_data, batch_size=batch_size)

    print(f'Training dataset size: {len(train_data)}')
    print(f'Test dataset size: {len(test_data)}')

    print(f'Using {device} device for training')

    model = AE(embedding_dim=embedding_dim).to(device)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = learning_rate,
        weight_decay = weight_decay
        )
    
    history = train(model, train_dl, test_dl, loss_fn, optimizer, epochs)
    wandb.finish()

    plt.plot(history['train_loss'], label='train')
    plt.plot(history['test_loss'], label='test')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss')
    plt.xticks(np.arange(0, epochs, 1))
    plt.show()

    sample_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    sample_batch = next(iter(sample_loader))
    sample_images, _ = sample_batch

    sample_images = sample_images.to(device)

    with torch.no_grad():
        reconstructed_images = model(sample_images)

    sample_images = sample_images.cpu().numpy()
    reconstructed_images = reconstructed_images.cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i in range(5):
        axes[0, i].imshow(sample_images[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_title('Original Images')
    axes[1, 0].set_title('Reconstructed Images')

    plt.show()

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'model_ae_{current_time}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an Autoencoder on FashionMNIST dataset")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of the embedding space')
    
    args = parser.parse_args()
    main(args.embedding_dim, args.batch_size, args.learning_rate, args.weight_decay, args.epochs, args.device)