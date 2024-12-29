import logging
import os
from logging import Formatter, StreamHandler, getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import VAE
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm

# hyperparameters
INPUT_DIM = 784
HIDDEN_DIM = 200
LATENT_DIM = 20
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
SAVE_INTERVAL = 5
WS_DIR = "src/vae/results"

# logging settings
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(handler_format)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(torch.flatten)]
)

dataset = datasets.MNIST(
    root="~/ML/dataset/image_datasets/mnist",
    train=True,
    download=True,
    transform=transform,
)


class VAETrainer:
    def __init__(
        self,
        ws_dir: str,
        train_dataset: data.Dataset,
        net: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        batch_size: int = 32,
        num_epochs: int = 30,
        save_interval: int = 5,
    ) -> None:
        self.net = net
        self.optimizer = optimizer
        self.device = device

        self.train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )

        self.ws_dir = ws_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_interval = save_interval

        logger.info("Trainer initialized")

    def train(self) -> list[dict]:
        self.net.to(self.device)
        # 学習と検証
        with tqdm(range(self.num_epochs), desc="Epoch") as t_global:
            train_loss: list[float] = []
            logs: list[dict] = []

            for epoch in t_global:
                # 学習
                with tqdm(self.train_dataloader, desc="Train", leave=False) as t:
                    sum_loss = 0.0  # lossの合計

                    for inputs, _ in t:
                        inputs = inputs.to(self.device)

                        self.optimizer.zero_grad()
                        loss = self.net.get_loss(inputs)
                        loss.backward()
                        self.optimizer.step()

                        loss = loss.item()
                        sum_loss += loss

                    loss = sum_loss / len(self.train_dataloader)
                    train_loss.append(loss)
                    t.set_postfix(train_loss=loss)

                t_global.set_postfix(train_loss=loss)

                logs.append(
                    {
                        "epoch": epoch,
                        "train_loss": np.mean(loss),
                    }
                )

                self.save_logs(logs)

                # モデルを保存
                if (epoch + 1) % self.save_interval == 0:
                    os.makedirs(os.path.join(self.ws_dir, "model"), exist_ok=True)
                    torch.save(
                        self.net.state_dict(),
                        os.path.join(
                            self.ws_dir, "model", "model_{}.pth".format(epoch + 1)
                        ),
                    )

        return logs

    def save_logs(self, logs: list) -> None:
        df = pd.DataFrame(logs)
        df.to_csv(os.path.join(self.ws_dir, "train_logs.csv"))

        plt.plot(df["train_loss"], label="train_loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Logs")
        plt.savefig(os.path.join(self.ws_dir, "train_logs.png"))
        plt.close()


if __name__ == "__main__":
    ws_dir = WS_DIR

    net = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = VAETrainer(ws_dir, dataset, net, optimizer, device)
    trainer.train()
