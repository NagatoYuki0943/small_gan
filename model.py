import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
torch.cuda

class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        padding = padding or dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.norm = norm(out_channels)
        if act is nn.ReLU:
            self.act = act(inplace=True)
        elif act is nn.LeakyReLU:
            self.act = act(0.2, inplace=True)
        else:
            self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


def test_conv():
    x = torch.randn(1, 3, 10, 10)
    model = ConvNormAct(in_channels=3, out_channels=4, kernel_size=3, dilation=2).eval()
    with torch.inference_mode():
        print(model(x).shape)   # [1, 4, 10, 10]


class TransposeConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.norm = norm(out_channels)
        if act is nn.ReLU:
            self.act = act(inplace=True)
        elif act is nn.LeakyReLU:
            self.act = act(0.2, inplace=True)
        else:
            self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


def test_transpose_conv():
    x = torch.randn(1, 3, 1, 1)
    model = TransposeConvNormAct(in_channels=3, out_channels=4, kernel_size=3, stride=1).eval()
    with torch.inference_mode():
        print(model(x).shape)   # [1, 4, 12, 12]

    model = TransposeConvNormAct(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1).eval()
    with torch.inference_mode():
        print(model(x).shape)   # [1, 4, 19, 19]

    model = TransposeConvNormAct(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1).eval()
    with torch.inference_mode():
        print(model(x).shape)   # [1, 4, 20, 20]

    model = TransposeConvNormAct(in_channels=3, out_channels=4, kernel_size=4, stride=2, padding=1).eval()
    with torch.inference_mode():
        print(model(x).shape)   # [1, 4, 20, 20]


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels, kernel_size, norm=norm, act=act),
            ConvNormAct(hidden_channels, in_channels, kernel_size, norm=norm, act=act)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(x)


class Generator(nn.Module):
    """上采样16倍的模型
    """
    def __init__(
        self,
        hidden_channels: int = 64,
        out_channels: int = 3,
        n_residual_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.stage1 = nn.Sequential(
                TransposeConvNormAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels//2,
                kernel_size=4,
                stride=2,
                padding=1,
                norm=nn.InstanceNorm2d
            ),
            *[ResidualBlock(in_channels=hidden_channels//2, norm=nn.InstanceNorm2d) for _ in range(n_residual_blocks)],
        )

        self.stage2 = nn.Sequential(
            TransposeConvNormAct(
                in_channels=hidden_channels//2,
                out_channels=hidden_channels//4,
                kernel_size=4,
                stride=2,
                padding=1,
                norm=nn.InstanceNorm2d
            ),
            *[ResidualBlock(hidden_channels//4, norm=nn.InstanceNorm2d) for _ in range(n_residual_blocks)],
        )

        self.stage3 = nn.Sequential(
            TransposeConvNormAct(
                in_channels=hidden_channels//4,
                out_channels=hidden_channels//8,
                kernel_size=4,
                stride=2,
                padding=1,
                norm=nn.InstanceNorm2d
            ),
            *[ResidualBlock(hidden_channels//8, norm=nn.InstanceNorm2d) for _ in range(n_residual_blocks)],
        )

        self.stage4 = nn.Sequential(
            TransposeConvNormAct(
                in_channels=hidden_channels//8,
                out_channels=hidden_channels//16,
                kernel_size=4,
                stride=2,
                padding=1,
                norm=nn.InstanceNorm2d
            ),
            *[ResidualBlock(hidden_channels//16, norm=nn.InstanceNorm2d) for _ in range(n_residual_blocks)],
        )

        self.stage5 = nn.Conv2d(hidden_channels//16, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = F.tanh(x)
        return x


def test_generater():
    x = torch.randn(2, 256, 7, 7)
    model = Generator(256, 1).eval()

    model.load_state_dict(torch.load('work_dirs/20240315-090721-mnist/latest_generator.pth'))
    with torch.inference_mode():
        print(model(x).shape)


class Discriminator(nn.Module):
    """下采样16倍的模型"""
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        n_residual_blocks: int = 1,
    ) -> None:
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels//8, kernel_size=3, stride=2, padding=1, norm=nn.InstanceNorm2d, act=nn.LeakyReLU),
            *[ResidualBlock(hidden_channels//8, norm=nn.InstanceNorm2d, act=nn.LeakyReLU) for i in range(n_residual_blocks)],
        )

        self.stage2 = nn.Sequential(
            ConvNormAct(hidden_channels//8, hidden_channels//4, kernel_size=3, stride=2, padding=1, norm=nn.InstanceNorm2d),
            *[ResidualBlock(hidden_channels//4, norm=nn.InstanceNorm2d, act=nn.LeakyReLU) for i in range(n_residual_blocks)],
        )

        self.stage3 = nn.Sequential(
            ConvNormAct(hidden_channels//4, hidden_channels//2, kernel_size=3, stride=2, padding=1, norm=nn.InstanceNorm2d),
            *[ResidualBlock(hidden_channels//2, norm=nn.InstanceNorm2d, act=nn.LeakyReLU) for i in range(n_residual_blocks)],
        )

        self.stage4 = nn.Sequential(
            ConvNormAct(hidden_channels//2, hidden_channels, kernel_size=3, stride=2, padding=1, norm=nn.InstanceNorm2d),
            *[ResidualBlock(hidden_channels, norm=nn.InstanceNorm2d, act=nn.LeakyReLU) for i in range(n_residual_blocks)],
        )

        self.stage5 = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(0),
        )

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.stage1(x)  # [B, 3, 28, 28] -> [B, 64, 14, 14]
        x = self.stage2(x)  # [B, 64, 14, 14] -> [B, 128, 7, 7]
        x = self.stage3(x)  # [B, 128, 7, 7] -> [B, 256, 4, 4]
        x = self.stage4(x)  # [B, 256, 4, 4] -> [B, 512, 2, 2]
        x = self.stage5(x)  # [B, 512, 2, 2] -> [B]
        x = F.sigmoid(x)
        return x


def test_discriminator():
    x = torch.randn(2, 3, 28, 28)
    model = Discriminator(3).eval()
    with torch.inference_mode():
        print(model(x).shape)


if __name__ == "__main__":
    # test_conv()
    # test_transpose_conv()
    test_generater()
    # test_discriminator()
