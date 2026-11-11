import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Tạo tập dữ liệu giả lập (Vì data_clean.csv của bạn có quá ít dòng để huấn luyện ML)
# Giả lập 100 điểm dữ liệu cảm biến
import numpy as np
np.random.seed(42)
data = {
    'Sensor_Temp': np.random.normal(25, 5, 100),
    'Sensor_Vibration': np.random.normal(10, 2, 100),
    # Giả định: Nhiệt độ > 30 và Rung > 12 thì cảnh báo lỗi (1), ngược lại là Bình thường (0)
}
df = pd.DataFrame(data)
df['Status'] = np.where((df['Sensor_Temp'] > 30) | (df['Sensor_Vibration'] > 12), 1, 0)

# 2. Chia tập Train và Test (Tỷ lệ 80% học - 20% thi)
X = df[['Sensor_Temp', 'Sensor_Vibration']] # Câu hỏi (Input)
y = df['Status']                            # Đáp án (Output)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Khởi tạo 3 mô hình thuật toán
models = {
    "Logistic Regression": LogisticRegression(),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# 4. Huấn luyện và Đánh giá (So sánh)
print("=== BẢNG SO SÁNH 3 MÔ HÌNH ML NỀN TẢNG ===")
print(f"{'Mô hình':<22} | {'Accuracy':<10} | {'F1-Score':<10}")
print("-" * 48)

for name, model in models.items():
    # Mô hình học (Train)
    model.fit(X_train, y_train)
    
    # Mô hình dự đoán (Test)
    predictions = model.predict(X_test)
    
    # Chấm điểm
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print(f"{name:<22} | {acc:.2f}       | {f1:.2f}")

print("-" * 48)