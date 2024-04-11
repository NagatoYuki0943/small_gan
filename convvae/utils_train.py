import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import math
import statistics
from tqdm import tqdm
from utils import remove_files


def loss_fn(X: Tensor, X_recon: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:

    # Reconstruction loss
    recon_loss = F.mse_loss(X_recon, X, reduction="sum")

    # KL divergence between encoder distribution and standard normal:
    kl_div = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp())

    # Total loss
    return recon_loss + kl_div


def train(
    save_dir: str,
    epochs: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler.LRScheduler,
    train_dataloader: DataLoader,
    device: str,
):
    print(f"save_dir: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "logs.csv", "w") as f:
        f.write("epoch, loss, lr\n")
    writer = SummaryWriter(save_dir)

    best_loss = math.inf
    best_loss = math.inf
    for epoch in range(1, epochs + 1):
        #---------- train ----------#
        pbar = tqdm(total=len(train_dataloader), desc=f"epoch:{epoch}-train")
        losses = []
        for images, _ in train_dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            x, mu, logvar = model(images)
            loss: Tensor = loss_fn(images, x, mu, logvar)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)

        # lr scheduler
        lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step()
        pbar.close()
        #---------- train ----------#

        #---------- save logs ----------#
        loss_mean = statistics.mean(losses)
        with open(save_dir / "logs.csv", "a") as f:
            f.write(f"{epoch}, {loss_mean}, {lr}\n")
        writer.add_scalar("loss", loss_mean, epoch)
        writer.add_scalar("lr", lr, epoch)
        #---------- save logs ----------#

        #---------- save latest and best model ----------#
        torch.save(model.state_dict(), save_dir / "latest_model.pth")
        torch.save(optimizer.state_dict(), save_dir / "latest_optimizer.pth")

        if best_loss > loss_mean:
            best_loss = loss_mean
            remove_files(save_dir, "best_")
            torch.save(model.state_dict(), save_dir / f"best_model_{epoch}_{loss_mean:.4f}.pth")
            torch.save(optimizer.state_dict(), save_dir / f"best_optimizer_{epoch}_{loss_mean:.4f}.pth")
        #---------- save latest and best model ----------#
