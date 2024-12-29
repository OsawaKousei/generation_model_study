import matplotlib.pyplot as plt
import torch
import torchvision
from model import VAE

INPUT_DIM = 784
HIDDEN_DIM = 200
LATENT_DIM = 20
MODEL_PATH = "src/vae/results/model/model_30.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))


with torch.no_grad():
    sample_size = 64
    z = torch.randn(sample_size, LATENT_DIM).to(device)
    x = net.decoder(z)
    generated_images = x.view(sample_size, 1, 28, 28).cpu()

    grid_img = torchvision.utils.make_grid(
        generated_images, nrow=8, padding=2, normalize=True
    )

    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
