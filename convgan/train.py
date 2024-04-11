import torch
from torch import nn, optim, Tensor
from pathlib import Path
import time

from model import Generator, Discriminator
from dataset import get_mnist, get_cifar
from utils import save_file
from utils_train import train


device = "cuda" if torch.cuda.is_available() else "cpu"

# hparam
batch_size = 128
epochs = 100
g_optimizer_lr = 0.001 # 0.0002
d_optimizer_lr = 0.001 # 0.0002
save_dir = Path(f"work_dirs/{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-mnist")

input_shape = [1, 32, 32]
hidden_channels = 64

# save train script
save_file(Path(__file__), save_dir)

# model
generator = Generator(input_shape=input_shape, hidden_channels=hidden_channels, n_residual_blocks=1).to(device)
discriminator = Discriminator(input_shape=input_shape, hidden_channels=hidden_channels).to(device)

# optimizer
# g_optimizer = optim.SGD(generator.parameters(), lr=g_optimizer_lr, momentum=0.937, weight_decay=5e-4)
g_optimizer = optim.Adam(generator.parameters(), lr=g_optimizer_lr)
g_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=epochs, eta_min=g_optimizer_lr * 0.1)
# d_optimizer = optim.SGD(discriminator.parameters(), lr=d_optimizer_lr, momentum=0.937, weight_decay=5e-4)
d_optimizer = optim.Adam(discriminator.parameters(), lr=d_optimizer_lr)
d_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=epochs, eta_min=d_optimizer_lr * 0.1)

# loss
loss_fn = nn.BCELoss()

# dataset
data_dir = "datasets"
train_dataloader, valid_dataloader = get_cifar(data_dir, batch_size=batch_size)


if __name__ == "__main__":
    train(
        save_dir=save_dir,
        epochs=epochs,
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g_lr_scheduler=g_lr_scheduler,
        d_lr_scheduler=d_lr_scheduler,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        device=device,
    )
