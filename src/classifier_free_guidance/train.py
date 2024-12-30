import logging
import os
from logging import Formatter, StreamHandler, getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from model import Diffuser, UNetCond
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm

# hyperparameters
IMG_SIZE = 28
BATCH_SIZE = 128
NUM_EPOCHS = 10
NUM_TIME_STEPS = 1000
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 10
WS_DIR = "src/classifier_free_guidance/results"

# logging settings
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(handler_format)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)


transform = transforms.ToTensor()
dataset = datasets.MNIST(
    root="~/ML/dataset/image_datasets/mnist",
    train=True,
    download=True,
    transform=transform,
)


class DiffusionModelTrainer:
    def __init__(
        self,
        ws_dir: str,
        train_dataset: data.Dataset,
        net: nn.Module,
        diffuser: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        batch_size: int = 32,
        num_epochs: int = 30,
        save_interval: int = 5,
    ) -> None:
        self.net = net
        self.diffuser = diffuser
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

                    for inputs, labels in t:
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                        self.optimizer.zero_grad()

                        time = torch.randint(
                            1, NUM_TIME_STEPS + 1, (len(inputs),), device=DEVICE
                        )
                        x_noisy, noise = self.diffuser.add_noise(inputs, time)

                        if np.random.random() < 0.1:
                            labels = None
                        noise_pred = self.net(x_noisy, time, labels)
                        loss = F.mse_loss(noise, noise_pred)
                        sum_loss += loss.item()

                        loss.backward()
                        self.optimizer.step()

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

    net = UNetCond(num_labels=NUM_LABELS)
    diffuser = Diffuser(device=DEVICE, num_time_steps=NUM_TIME_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = DiffusionModelTrainer(
        ws_dir, dataset, net, diffuser, optimizer, device, BATCH_SIZE, NUM_EPOCHS
    )
    trainer.train()
