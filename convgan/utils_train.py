import torch
from  torch import nn, optim, Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import math
import statistics
from tqdm import tqdm
from utils import remove_files


def train(
    save_dir: str,
    epochs: int,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: optim.Optimizer,
    d_optimizer: optim.Optimizer,
    g_lr_scheduler: optim.lr_scheduler.LRScheduler,
    d_lr_scheduler: optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    train_dataloader: DataLoader,
    device: str,
):
    print(f"save_dir: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "logs.csv", "w") as f:
        f.write("epoch, g_loss, d_loss, generator_lr, discriminator_lr\n")
    writer = SummaryWriter(save_dir)

    best_g_loss = math.inf
    best_d_loss = math.inf
    for epoch in range(1, epochs + 1):
        #---------- train ----------#
        pbar = tqdm(total=len(train_dataloader), desc=f"epoch:{epoch}-train")
        g_losses = []
        d_losses = []
        for real_images, _ in train_dataloader:
            real_images = real_images.to(device)
            b, c, h, w = real_images.shape
            real_targets = torch.ones(b).to(device)
            fake_targets = torch.zeros(b).to(device)
            fake_input: Tensor = torch.randn(b, c, h, w).to(device)

            # train generator
            g_optimizer.zero_grad()
            fake_images: Tensor = generator(fake_input)
            fake_scores: Tensor = discriminator(fake_images)
            g_loss: Tensor = loss_fn(fake_scores, real_targets)
            g_loss.backward()
            g_optimizer.step()

            # train discriminator
            d_optimizer.zero_grad()
            real_scores: Tensor = discriminator(real_images)
            d_loss1: Tensor = loss_fn(real_scores, real_targets)
            fake_scores: Tensor = discriminator(fake_images.detach()) # detach the gradient
            d_loss2: Tensor = loss_fn(fake_scores, fake_targets)
            d_loss = (d_loss1 + d_loss2) / 2
            d_loss.backward()
            d_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            pbar.set_postfix({"g_loss": g_loss.item(), "d_loss": d_loss.item()})
            pbar.update(1)

        # lr scheduler
        g_lr = g_optimizer.param_groups[0]["lr"]
        d_lr = d_optimizer.param_groups[0]["lr"]
        g_lr_scheduler.step()
        d_lr_scheduler.step()
        pbar.close()
        #---------- train ----------#

        #---------- save logs ----------#
        g_loss_mean = statistics.mean(g_losses)
        d_loss_mean = statistics.mean(d_losses)
        with open(save_dir / "logs.csv", "a") as f:
            f.write(f"{epoch}, {g_loss_mean}, {d_loss_mean}, {g_lr}, {d_lr}\n")
        writer.add_scalar("g_loss", g_loss_mean, epoch)
        writer.add_scalar("d_loss", d_loss_mean, epoch)
        writer.add_scalar("g_lr", g_lr, epoch)
        writer.add_scalar("d_lr", d_lr, epoch)
        #---------- save logs ----------#

        #---------- save latest and best model ----------#
        torch.save(generator.state_dict(), save_dir / "latest_generator.pth")
        torch.save(g_optimizer.state_dict(), save_dir / "latest_generator_optimizer.pth")
        torch.save(discriminator.state_dict(), save_dir / "latest_discriminator.pth")
        torch.save(d_optimizer.state_dict(), save_dir / "latest_discriminator_optimizer.pth")

        if best_g_loss > g_loss_mean:
            best_g_loss = g_loss_mean
            remove_files(save_dir, "best_generator")
            torch.save(generator.state_dict(), save_dir / f"best_generator_{epoch}_{g_loss_mean:.4f}.pth")
            torch.save(g_optimizer.state_dict(), save_dir / f"best_generator_optimizer_{epoch}_{g_loss_mean:.4f}.pth")
        if best_d_loss > d_loss_mean:
            best_d_loss = d_loss_mean
            remove_files(save_dir, "best_discriminator")
            torch.save(discriminator.state_dict(), save_dir / f"best_discriminator_{epoch}_{d_loss_mean:.4f}.pth")
            torch.save(d_optimizer.state_dict(), save_dir / f"best_discriminator_optimizer_{epoch}_{d_loss_mean:.4f}.pth")
        #---------- save latest and best model ----------#
