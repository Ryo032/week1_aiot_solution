import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu sạch từ bài thực hành Python trước
df = pd.read_csv("data_clean.csv")

print("=== TÍNH TOÁN NHANH VỚI NUMPY ===")
values = df["Sensor_Value"].to_numpy()
print(f"Nhiệt độ trung bình: {np.mean(values):.2f} °C")
print(f"Độ lệch chuẩn: {np.std(values):.2f} °C")

# 2. Vẽ biểu đồ trực quan hóa (EDA) với Matplotlib
plt.figure(figsize=(10, 4))

# Biểu đồ 1: Xu hướng nhiệt độ cảm biến theo thời gian
plt.subplot(1, 2, 1)
plt.plot(df["Timestamp"], df["Sensor_Value"], marker="o", color="tab:blue", linewidth=2)
plt.axhline(np.mean(values), color="red", linestyle="--", label="Trung bình")
plt.title("Biến động giá trị cảm biến")
plt.xlabel("Thời gian (Timestamp)")
plt.ylabel("Sensor Value")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

# Biểu đồ 2: Thống kê số lượng trạng thái thiết bị (Status)
plt.subplot(1, 2, 2)
status_counts = df["Status"].value_counts()
plt.bar(status_counts.index, status_counts.values, color=["tab:green", "tab:orange"])
plt.title("Tần suất trạng thái (Status)")
plt.xlabel("Trạng thái")
plt.ylabel("Số lần ghi nhận")

plt.tight_layout()
plt.savefig("eda_report.png", dpi=150)
print("\n=== ĐÃ XUẤT BIỂU ĐỒ THÀNH CÔNG: eda_report.png ===")