import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim

# 1. Tạo dữ liệu cảm biến giả lập (500 dòng) & Feature Engineering (Tuần 3)
np.random.seed(42)
torch.manual_seed(42)

df = pd.DataFrame({
    "Temp": np.random.normal(28, 6, 500),
    "Vibration": np.random.normal(12, 3, 500),
    "Humidity": np.random.normal(60, 10, 500)
})
# Feature Engineering: Tạo cột đặc trưng mới (Tỷ lệ Nhiệt / Độ ẩm)
df["Heat_Stress_Index"] = df["Temp"] / (df["Humidity"] + 1e-5)

# Nhãn thực tế: Cảnh báo (1) nếu quá nhiệt kèm rung mạnh, ngược lại (0)
df["Status"] = np.where((df["Temp"] + df["Vibration"] * 0.8) > 39.0, 1, 0)

X = df[["Temp", "Vibration", "Humidity", "Heat_Stress_Index"]].values
y = df["Status"].values

# Chuẩn hóa dữ liệu (Bắt buộc cho PCA, K-Means và Mạng nơ-ron MLP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================================
# PHẦN 1: HỌC KHÔNG GIÁM SÁT & MÔ HÌNH MẠNH (TUẦN 3)
# ==========================================================
# Giảm chiều từ 4 cột xuống 2 cột bằng PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Phân cụm 2 nhóm không cần nhãn bằng KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
print(f"[TUẦN 3 - UNSUPERVISED] Đã giảm chiều PCA {X_pca.shape} & phân cụm K-Means: {np.bincount(clusters)}")

# Chia tập Train/Test cho Học có giám sát
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

ml_models = {
    "Random Forest (Song song)": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost (Nối tiếp sửa sai)": XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric="logloss", random_state=42)
}

print("\n=== BẢNG SO SÁNH MÔ HÌNH MẠNH TUẦN 3 & DEEP LEARNING MLP TUẦN 4 ===")
print(f"{'Mô hình':<28} | {'Accuracy':<10} | {'F1-Score':<10}")
print("-" * 55)

for name, model in ml_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"{name:<28} | {accuracy_score(y_test, preds):.2f}       | {f1_score(y_test, preds):.2f}")

# ==========================================================
# PHẦN 2: MẠNG NƠ-RON ĐA LỚP - MLP BẰNG PYTORCH (TUẦN 4)
# ==========================================================
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)

class SensorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

mlp_model = SensorMLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)

# Vòng lặp huấn luyện 100 kỷ nguyên (Epochs)
mlp_model.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = mlp_model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# Đánh giá mạng MLP trên tập Test
mlp_model.eval()
with torch.no_grad():
    mlp_preds = (mlp_model(X_test_t).numpy() > 0.5).astype(int).flatten()
    print(f"{'PyTorch MLP (Deep Learning)':<28} | {accuracy_score(y_test, mlp_preds):.2f}       | {f1_score(y_test, mlp_preds):.2f}")
print("-" * 55)