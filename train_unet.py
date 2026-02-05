import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

# ============================
# Dataset
# ============================

class FloorPlanDataset(Dataset):
    def __init__(self, root, image_size=512):
        self.images = sorted(glob.glob(os.path.join(root, "floorplan_*.png")))
        self.masks  = sorted(glob.glob(os.path.join(root, "mask_*.png")))
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = imageio.imread(self.images[idx])
        mask = imageio.imread(self.masks[idx])

        img = self.transform_img(img)
        mask = torch.from_numpy(mask).long()

        return img, mask

# ============================
# Simple U-Net model
# ============================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.middle = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool1(c1))
        c3 = self.down3(self.pool2(c2))
        c4 = self.down4(self.pool3(c3))
        c5 = self.middle(self.pool4(c4))
        u4 = self.up4(c5)
        u4 = torch.cat([u4, c4], dim=1)
        u4 = self.conv4(u4)
        u3 = self.up3(u4)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.conv3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.conv2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.conv1(u1)
        out = self.outc(u1)
        return out

# ============================
# Training
# ============================

def train_model(data_dir, n_classes, epochs=10, batch_size=2, lr=1e-4, device="cuda"):
    dataset = FloorPlanDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = UNet(n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}: loss = {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "unet_floorplan.pth")
    print("Model saved.")

# ============================
# Run training
# ============================

if __name__ == "__main__":
    data_dir = "data"
    n_classes = 101  # 100 rooms + background
    train_model(data_dir, n_classes, epochs=10)
