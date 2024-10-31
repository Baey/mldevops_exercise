import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder model for an autoencoder neural network.

    This model consists of convolutional layers followed by batch normalization
    and max pooling layers, and then fully connected layers to produce an embedding.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization for the first convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization for the second convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        fc1 (nn.Linear): First fully connected layer.
        bn3 (nn.BatchNorm1d): Batch normalization for the first fully connected layer.
        fc2 (nn.Linear): Second fully connected layer that outputs the embedding.

    Args:
        embedding_dim (int): Dimension of the output embedding.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the encoder. Takes an input tensor `x` and returns the encoded embedding.
    """
    ''''''
    def __init__(self, embedding_dim: int):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = nn.Flatten()(x)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    """
    Decoder neural network for an autoencoder model.
    This class defines the decoder part of an autoencoder, which takes an embedding vector
    and reconstructs it back to the original input shape.
    Attributes:
        fc1 (nn.Linear): Fully connected layer to transform the embedding dimension to 128 units.
        bn4 (nn.BatchNorm1d): Batch normalization layer for the output of fc1.
        fc2 (nn.Linear): Fully connected layer to transform 128 units to 64 * 7 * 7 units.
        deconv1 (nn.ConvTranspose2d): Transposed convolution layer to upsample from 64 channels to 64 channels.
        bn5 (nn.BatchNorm2d): Batch normalization layer for the output of deconv1.
        deconv2 (nn.ConvTranspose2d): Transposed convolution layer to upsample from 64 channels to 32 channels.
        bn6 (nn.BatchNorm2d): Batch normalization layer for the output of deconv2.
        conv1 (nn.Conv2d): Convolution layer to transform 32 channels to 1 channel.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the decoder. Takes an embedding tensor and reconstructs
            it to the original input shape.
    """
    
    def __init__(self, embedding_dim: int):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.bn5(self.deconv1(x)))
        x = F.relu(self.bn6(self.deconv2(x)))
        x = torch.sigmoid(self.conv1(x))
        return x

class Autoencoder(nn.Module):
    """
    Autoencoder (AE) model.
    This model consists of an encoder and a decoder. The encoder compresses the input data into a lower-dimensional
    representation (embedding), and the decoder reconstructs the original data from this embedding.
    Attributes:
        encoder (nn.Module): The encoder part of the autoencoder.
        decoder (nn.Module): The decoder part of the autoencoder.
    Args:
        embedding_dim (int): The dimensionality of the embedding space.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the autoencoder, encoding the input and then decoding it.
    """
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.decoder = Decoder(embedding_dim=embedding_dim)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded