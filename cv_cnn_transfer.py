import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# 1. Tạo Dataset ảnh AIoT (Đọc từ images_raw hoặc tạo ảnh mẫu 64x64)
class AIoTImageDataset(Dataset):
    def __init__(self, transform=None, num_samples=32):
        self.transform = transform
        self.images = []
        self.labels = []
        raw_dir = "images_raw"
        img_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(raw_dir) else []

        for i in range(num_samples):
            if img_files:
                img = Image.open(img_files[i % len(img_files)]).convert("RGB")
            else:
                arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
            self.images.append(img)
            self.labels.append(i % 2) # Nhãn 0: OK, Nhãn 1: Lỗi

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# 2. Tăng cường dữ liệu - Data Augmentation (Tuần 4)
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

dataset = AIoTImageDataset(transform=train_transform, num_samples=32)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ==========================================================
# PHẦN 1: MẠNG TÍCH CHẬP CNN TỰ XÂY DỰNG (TUẦN 3)
# ==========================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Giảm kích thước từ 64x64 -> 32x32
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Giảm kích thước từ 32x32 -> 16x16
        )
        self.classifier = nn.Linear(16 * 16 * 16, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ==========================================================
# PHẦN 2: HỌC CHUYỂN GIAO - FINE-TUNE RESNET18 (TUẦN 4)
# ==========================================================
def build_finetuned_resnet():
    model = models.resnet18(weights=None)
    # Đóng băng (Freeze) toàn bộ các tầng trích xuất đặc trưng đầu
    for param in model.parameters():
        param.requires_grad = False
    # Thay tầng phân loại cuối (model.fc) bằng tầng Linear mới cho 2 lớp
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

# Hàm huấn luyện nhanh 3 Epochs và in đường cong Loss (Learning Curve)
def train_quick(model, name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)
    model.train()
    losses = []
    for epoch in range(3):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(round(epoch_loss / len(train_loader), 4))
    print(f"[{name:<25}] Đường cong Train Loss qua 3 Epochs: {losses}")

print("=== KẾT QUẢ HUẤN LUYỆN CNN (TUẦN 3) & FINE-TUNE RESNET18 (TUẦN 4) ===")
train_quick(SimpleCNN(), "Tuần 3: Simple CNN")
train_quick(build_finetuned_resnet(), "Tuần 4: ResNet18 Transfer")