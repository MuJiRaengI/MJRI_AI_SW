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
import numpy as np
from PIL import Image
import torch

from source.ai.rl.model.resnet import Resnet18
from source.envs.env_avoid_observber import EnvAvoidObserver
from source.ai.rl.model.find_avoid_observer_model import AutoEncoder
from torch.utils.data import Dataset, Subset


class CustomImageDataset(Dataset):
    def __init__(self, length, fixed_seed=None):
        self.env = EnvAvoidObserver(
            num_observers=500, random_start=True, move_observer=True, frame_stack=5
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
        # labels = stacks[12:]
        return inputs, stacks


def train():
    train_dataset = CustomImageDataset(length=1000)
    val_dataset = CustomImageDataset(length=100, fixed_seed=42)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)
    model.load_state_dict(
        torch.load(
            r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\results_ae_full_size\autoencoder_best.pth",
            map_location=device,
        )
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    save_dir = os.path.join(os.path.abspath("."), "results_ae")
    print(f"Saving results to: {save_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    epochs = 1000
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
        first_saved = False
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Valid Epoch {epoch+1}/{epochs}"):
                x = x.to(device)
                y = y.to(device)
                x_hat, _ = model(x)
                loss = criterion(x_hat, y)
                val_loss += loss.item() * x.size(0)
                # 첫 번째 배치의 첫 번째 샘플을 이미지로 저장
                if not first_saved:
                    diff_npy = torch.abs(x_hat[0].detach().cpu() - y[0].detach().cpu())
                    diff_npy = (
                        torch.clip(diff_npy, 0, 1).permute(1, 2, 0).numpy() * 255
                    ).astype(np.uint8)
                    pred_npy = tensor_to_npy(torch.clip(x_hat[0], 0, 1).unsqueeze(0))
                    label_npy = tensor_to_npy(torch.clip(y[0], 0, 1).unsqueeze(0))
                    for i in range(pred_npy.shape[2] // 3):
                        pred_img = Image.fromarray(pred_npy[:, :, i * 3 : (i + 1) * 3])
                        pred_img.save(
                            os.path.join(save_dir, f"pred_epoch_{epoch+1}_{i}.png")
                        )
                        # save diff image
                        diff_img = Image.fromarray(diff_npy[:, :, i * 3 : (i + 1) * 3])
                        diff_img.save(
                            os.path.join(save_dir, f"diff_epoch_{epoch+1}_{i}.png")
                        )

                        if epoch == 0:
                            label_img = Image.fromarray(
                                label_npy[:, :, i * 3 : (i + 1) * 3]
                            )
                            label_img.save(
                                os.path.join(save_dir, f"label_epoch_{epoch+1}_{i}.png")
                            )
                    first_saved = True
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # Best model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            # torch.save(model.state_dict(), "autoencoder_best.pth")
            model_path = os.path.join(save_dir, "autoencoder_best.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f"[Best] Model saved at epoch {epoch+1} (Val Loss: {avg_val_loss:.6f})"
            )

    # 마지막 모델도 저장
    # torch.save(model.state_dict(), "autoencoder_last.pth")
    model_path = os.path.join(save_dir, "autoencoder_last.pth")
    torch.save(model.state_dict(), model_path)
    print(
        f"Autoencoder 학습 완료! Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f}"
    )


def eval():
    test_dataset = CustomImageDataset(length=100, fixed_seed=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)

    model_path = (
        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\results_ae\autoencoder_best.pth"
    )
    # model_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\pretrained\avoid_observer\autoencoder_tyndall_log.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    # test
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"test"):
            x = x.to(device)
            y = y.to(device)

            pred, _ = model(x)
            # pp = torch.clip(pred[0], 0, 1)
            # yy = y[0]
            pred_npy = tensor_to_npy(torch.clip(pred[0], 0, 1).unsqueeze(0))
            label_npy = tensor_to_npy(torch.clip(y[0], 0, 1).unsqueeze(0))
            diff_npy = np.abs(pred_npy - label_npy)

            for i in range(pred_npy.shape[2] // 3):
                pred_img = Image.fromarray(pred_npy[:, :, i * 3 : (i + 1) * 3])
                pred_img.save(
                    os.path.join(
                        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW", f"pred_{i}.png"
                    )
                )
                label_img = Image.fromarray(label_npy[:, :, i * 3 : (i + 1) * 3])
                label_img.save(
                    os.path.join(
                        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW", f"label_{i}.png"
                    )
                )
                diff_img = Image.fromarray(diff_npy[:, :, i * 3 : (i + 1) * 3])
                diff_img.save(
                    os.path.join(
                        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW", f"diff_{i}.png"
                    )
                )
            print()


def tensor_to_image(tensor):
    """
    4D tensor (B, C, H, W) → PIL.Image (첫 번째 배치만)
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    img = tensor[0].detach().numpy()  # (3, H, W)
    img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
    img = np.clip(img, 0, 1)  # 값이 0~1 범위라면
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def tensor_to_npy(tensor):
    """
    4D tensor (B, C, H, W) → PIL.Image (첫 번째 배치만)
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    img = tensor[0].detach().numpy()  # (3, H, W)
    img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
    img = np.clip(img, 0, 1)  # 값이 0~1 범위라면
    img = (img * 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    train()
    pygame.quit()
    # eval()

    # autoencoder 학습 후에 reset() 메서드 수정
