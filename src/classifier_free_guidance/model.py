import math

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm


def _positional_encoding(
    t: torch.Tensor, output_dim: int, device: torch.device
) -> torch.Tensor:
    D = output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])

    return v


def positional_encoding(
    ts: torch.Tensor, output_dim: int, device: torch.device
) -> torch.Tensor:
    batch_size = len(ts)
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _positional_encoding(ts[i], output_dim, device)
    return v


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super(ConvBlock, self).__init__()
        self.covs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
        )

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        N, C, _, _ = x.shape
        v = self.mlp(v).view(N, C, 1, 1)
        y = self.covs(x + v)
        return y


class UNetCond(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        time_embed_dim: int = 100,
        num_labels: int | None = None,
    ):
        super(UNetCond, self).__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_channels, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(64 + 128, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_channels, 1)

        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        if num_labels is not None:
            self.label_embed = nn.Embedding(num_labels, time_embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_steps: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v = positional_encoding(time_steps, self.time_embed_dim, x.device)

        if labels is not None:
            v += self.label_embed(labels)

        x1 = self.down1(x, v)
        x2 = self.down2(self.max_pool(x1), v)
        x3 = self.bot1(self.max_pool(x2), v)
        x = self.upsample(x3)
        x = self.up2(torch.cat([x2, x], dim=1), v)
        x = self.upsample(x)
        x = self.up1(torch.cat([x1, x], dim=1), v)
        x = self.out(x)
        return x


class Diffuser:
    def __init__(
        self,
        device: torch.device,
        num_time_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_time_steps = num_time_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_time_steps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert (t >= 1).all() and (t <= self.num_time_steps).all()
        t_idx = t.long() - 1

        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        gamma: float = 3.0,
    ) -> torch.Tensor:
        assert (t >= 1).all() and (t <= self.num_time_steps).all()
        t_idx = t.long() - 1

        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_prev = self.alphas[t_idx - 1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_prev = alpha_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps_cond = model(x, t, labels)
            eps_no_cond = model(x, t)
            eps = eps_no_cond + gamma * (eps_cond - eps_no_cond)
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0

        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_prev) / (1 - alpha_bar))

        return mu + std * noise

    def reverse_to_img(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 255
        x = x.clamp(0, 255).to(torch.uint8).cpu()
        to_pil = torchvision.transforms.ToPILImage()
        return to_pil(x)

    def sample(
        self,
        model: nn.Module,
        x_shape: tuple = (20, 1, 28, 28),
        labels: torch.Tensor | None = None,
    ) -> list:
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_time_steps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]

        return images
