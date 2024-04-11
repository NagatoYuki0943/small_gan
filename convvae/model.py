# Copyright © 2023-2024 Apple Inc.

import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class UpsamplingConv2d(nn.Module):
    """
    A convolutional layer that upsamples the input by a factor of 2. MLX does
    not yet support transposed convolutions, so we approximate them with
    nearest neighbor upsampling followed by a convolution. This is similar to
    the approach used in the original U-Net.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))
        return x


class Encoder(nn.Module):
    """
    A convolutional variational encoder.
    Maps the input to a normal distribution in latent space and sample a latent
    vector from that distribution.
    """

    def __init__(self, num_latent_dims, image_shape, max_num_dims):
        super().__init__()

        # number of filters in the convolutional layers
        num_dims_1 = max_num_dims // 4
        num_dims_2 = max_num_dims // 2
        num_dims_3 = max_num_dims

        # Output (BHWC):  B x 32 x 32 x num_dims_1
        self.conv1 = nn.Conv2d(image_shape[0], num_dims_1, 3, stride=2, padding=1)
        # Output (BHWC):  B x 16 x 16 x num_dims_2
        self.conv2 = nn.Conv2d(num_dims_1, num_dims_2, 3, stride=2, padding=1)
        # Output (BHWC):  B x 8 x 8 x num_dims_3
        self.conv3 = nn.Conv2d(num_dims_2, num_dims_3, 3, stride=2, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(num_dims_1)
        self.bn2 = nn.BatchNorm2d(num_dims_2)
        self.bn3 = nn.BatchNorm2d(num_dims_3)

        # Divide the spatial dimensions by 8 because of the 3 strided convolutions
        output_shape = [num_dims_3] + [
            dimension // 8 for dimension in image_shape[1:]
        ]

        flattened_dim = math.prod(output_shape)

        # Linear mappings to mean and standard deviation
        self.proj_mu = nn.Linear(flattened_dim, num_latent_dims)
        self.proj_log_var = nn.Linear(flattened_dim, num_latent_dims)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        mu = self.proj_mu(x)
        logvar = self.proj_log_var(x)
        # Ensure this is the std deviation, not variance
        sigma = torch.exp(logvar * 0.5)

        # Generate a tensor of random values from a normal distribution
        eps = torch.randn(sigma.shape).to(x.device)

        # Reparametrization trick to brackpropagate through sampling.
        z = eps * sigma + mu

        return z, mu, logvar


class Decoder(nn.Module):
    """A convolutional decoder"""

    def __init__(self, num_latent_dims, image_shape, max_num_dims):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        num_img_channels = image_shape[0]
        self.max_num_dims = max_num_dims

        # decoder layers
        num_dims_1 = max_num_dims
        num_dims_2 = max_num_dims // 2
        num_dims_3 = max_num_dims // 4

        # divide the last two dimensions by 8 because of the 3 upsampling convolutions
        self.input_shape = [dimension // 8 for dimension in image_shape[1:]] + [
            num_dims_1
        ]
        flattened_dim = math.prod(self.input_shape)

        # Output: flattened_dim
        self.lin1 = nn.Linear(num_latent_dims, flattened_dim)
        # Output (BHWC):  B x 16 x 16 x num_dims_2
        self.upconv1 = UpsamplingConv2d(
            num_dims_1, num_dims_2, 3, stride=1, padding=1
        )
        # Output (BHWC):  B x 32 x 32 x num_dims_1
        self.upconv2 = UpsamplingConv2d(
            num_dims_2, num_dims_3, 3, stride=1, padding=1
        )
        # Output (BHWC):  B x 64 x 64 x #img_channels
        self.upconv3 = UpsamplingConv2d(
            num_dims_3, num_img_channels, 3, stride=1, padding=1
        )

        # Batch Normalizations
        self.bn1 = nn.BatchNorm2d(num_dims_2)
        self.bn2 = nn.BatchNorm2d(num_dims_3)

    def forward(self, z: Tensor) -> Tensor:
        x = self.lin1(z)

        # reshape to BHWC
        x = x.reshape(
            -1, self.max_num_dims, self.input_shape[0], self.input_shape[1]
        )

        # approximate transposed convolutions with nearest neighbor upsampling
        x = F.leaky_relu(self.bn1(self.upconv1(x)))
        x = F.leaky_relu(self.bn2(self.upconv2(x)))
        # sigmoid to ensure pixel values are in [0,1]
        x = torch.sigmoid(self.upconv3(x))
        return x


class CVAE(nn.Module):
    """
    A convolutional variational autoencoder consisting of an encoder and a
    decoder.
    """

    def __init__(self, num_latent_dims, input_shape, max_num_dims):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.encoder = Encoder(num_latent_dims, input_shape, max_num_dims)
        self.decoder = Decoder(num_latent_dims, input_shape, max_num_dims)

    def forward(self, x: Tensor) -> Tensor:
        # image to latent vector
        z, mu, logvar = self.encoder(x)
        # latent vector to image
        x = self.decode(z)
        return x, mu, logvar

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)[0]

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)


if __name__ == "__main__":
    model = CVAE(128, (3, 64, 64), 64).eval()
    x = torch.zeros(1, 3, 64, 64)
    with torch.inference_mode():
        x, mu, logvar = model(x)
        print(x.shape, mu.shape, logvar.shape)
        z = model.encode(x)
        print(z.shape)
        x = model.decode(z)
        print(x.shape)
