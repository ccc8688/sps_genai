import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sps_genai.generative_models import EnergyModel, DiffusionModel

def get_cifar10_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

def train_diffusion(loader, epochs=3, lr=2e-4, device="cpu"):
    print(f"--- Training Diffusion Model ({device}) ---")
    model = DiffusionModel(timesteps=200).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss = model.loss(images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Diffusion] Epoch {epoch}/{epochs} | MSE Loss: {total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "diffusion_cifar10.pth")
    print("Saved diffusion_cifar10.pth")

def train_ebm(loader, epochs=3, lr=1e-4, device="cpu"):
    print(f"--- Training Energy-Based Model ({device}) ---")
    model = EnergyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for real_images, _ in loader:
            real_images = real_images.to(device)
            fake_images = model.sample(batch_size=real_images.size(0), steps=20, device=device)
            model.train()
            optimizer.zero_grad()
            real_energy = model(real_images)
            fake_energy = model(fake_images)
            loss = (real_energy.mean() - fake_energy.mean()) + 0.1 * (real_energy.pow(2).mean() + fake_energy.pow(2).mean())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[EBM] Epoch {epoch}/{epochs} | Loss: {total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "ebm_cifar10.pth")
    print("Saved ebm_cifar10.pth")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    loader = get_cifar10_loader(batch_size=128)
    train_diffusion(loader, epochs=3, device=device)
    train_ebm(loader, epochs=3, device=device)
