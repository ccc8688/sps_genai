import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EnergyModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, 2, 1), Swish(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, 2, 1), Swish(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, 2, 1), Swish(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, 2, 1), Swish(),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4 * 2 * 2, hidden_dim * 2), Swish(),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def sample(self, batch_size=1, steps=60, step_size=10.0, noise_scale=0.005, device="cpu"):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        x = torch.empty((batch_size, 3, 32, 32), device=device).uniform_(-1.0, 1.0)
        x.requires_grad_(True)
        for _ in range(steps):
            noise = torch.randn_like(x) * noise_scale
            x_noisy = (x + noise).detach().requires_grad_(True)
            energy = self.forward(x_noisy).sum()
            grad = torch.autograd.grad(energy, x_noisy)[0]
            grad = torch.clamp(grad, -0.03, 0.03)
            with torch.no_grad():
                x = torch.clamp(x_noisy - step_size * grad, -1.0, 1.0)
        for p in self.parameters():
            p.requires_grad_(True)
        return x.detach()

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.gn1(self.conv1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.shortcut(x)

class SimpleUNet(nn.Module):
    def __init__(self, base_ch=64, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.inc = ResBlock(3, base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 3, 2, 1)
        self.enc1 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, 2, 1)
        self.enc2 = ResBlock(base_ch * 2, base_ch * 4, time_dim)
        self.bot = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec1 = ResBlock(base_ch * 4, base_ch * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec2 = ResBlock(base_ch * 2, base_ch, time_dim)
        self.outc = nn.Conv2d(base_ch, 3, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        x1 = self.inc(x, t_emb)
        x2 = self.enc1(self.down1(x1), t_emb)
        x3 = self.enc2(self.down2(x2), t_emb)
        b = self.bot(x3, t_emb)
        d1 = self.dec1(torch.cat([self.up1(b), x2], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d1), x1], dim=1), t_emb)
        return self.outc(d2)

class DiffusionModel(nn.Module):
    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        self.unet = SimpleUNet()
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def loss(self, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        xt = self.sqrt_alphas_cumprod[t][:, None, None, None] * x0 + self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
        return F.mse_loss(self.unet(xt, t), noise)

    @torch.no_grad()
    def sample(self, batch_size=1, device="cpu"):
        self.eval()
        x = torch.randn((batch_size, 3, 32, 32), device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t)
            alpha, alpha_hat, beta = self.alphas[i], self.alphas_cumprod[i], self.betas[i]
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = (1.0 / torch.sqrt(alpha)) * (x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_hat)) * pred_noise) + torch.sqrt(beta) * noise
        return torch.clamp(x, -1.0, 1.0)
