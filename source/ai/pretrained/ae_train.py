import os
import sys

sys.path.append(os.path.abspath("."))


import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from source.ai.rl.model.resnet import Resnet18
from source.envs.env_avoid_observber import EnvAvoidObserver
from torch.utils.data import Dataset, Subset


class CustomImageDataset(Dataset):
    def __init__(self, length, fixed_seed=None):
        self.env = EnvAvoidObserver(
            num_observers=100, random_start=True, move_observer=True, frame_stack=5
        )
        self.length = length
        self.fixed_seed = fixed_seed
        pygame.init()

    def transform(self, data):
        return data / 255.0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Validation용이면 seed 고정
        if self.fixed_seed is not None:
            np.random.seed(self.fixed_seed + idx)
            obs = self.env.reset(seed=self.fixed_seed + idx)
        else:
            obs = self.env.reset()
        action = np.random.randint(0, 8)
        for i in range(5):
            self.env.step(action)
        stacks = self.transform(self.env.get_stacked_state())
        inputs = stacks[:12]
        labels = stacks[12:]
        return inputs, labels


# 간단한 오토인코더 모델 정의
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(
            12, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out, feat


def train():
    ae = AutoEncoder()
    ae(torch.randn(1, 12, 256, 256))  # 예시 입력

    train_dataset = CustomImageDataset(length=1000)
    val_dataset = CustomImageDataset(length=100, fixed_seed=42)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    epochs = 100
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = criterion(x_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Valid Epoch {epoch+1}/{epochs}"):
                x = x.to(device)
                y = y.to(device)
                x_hat, _ = model(x)
                loss = criterion(x_hat, y)
                val_loss += loss.item() * x.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # Best model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "autoencoder_best.pth")
            print(
                f"[Best] Model saved at epoch {epoch+1} (Val Loss: {avg_val_loss:.6f})"
            )

    # 마지막 모델도 저장
    torch.save(model.state_dict(), "autoencoder_last.pth")
    print(
        f"Autoencoder 학습 완료! Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f}"
    )


if __name__ == "__main__":
    train()
    pygame.quit()
