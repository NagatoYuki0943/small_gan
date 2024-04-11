import torch
from torch import nn, optim, Tensor
from pathlib import Path
import time

from model import CVAE
from dataset import get_mnist, get_cifar
from utils import save_file
from utils_train import train


device = "cuda" if torch.cuda.is_available() else "cpu"

# hparam
batch_size = 128
epochs = 1000
optimizer_lr = 0.001 # 0.0002
save_dir = Path(f"work_dirs/{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-mnist")

num_latent_dims = 128
max_num_dims = 64
input_shape = [1, 32, 32]

# save train script
save_file(Path(__file__), save_dir)

# model
model = CVAE(num_latent_dims, input_shape, max_num_dims).to(device)

# optimizer
# optimizer = optim.SGD(model.parameters(), lr=optimizer_lr, momentum=0.937, weight_decay=5e-4)
optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=optimizer_lr * 0.1)

# dataset
data_dir = "datasets"
train_dataloader, valid_dataloader = get_mnist(data_dir, batch_size=batch_size)


if __name__ == "__main__":
    train(
        save_dir=save_dir,
        epochs=epochs,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        device=device,
    )
