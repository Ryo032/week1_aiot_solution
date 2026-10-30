import os
import pandas as pd

# Đường dẫn file dữ liệu thô và file sau khi làm sạch
INPUT_FILE = "data_raw.csv"
OUTPUT_FILE = "data_clean.csv"

def clean_and_analyze_csv(input_path, output_path):
    print("=== BẮT ĐẦU QUY TRÌNH LÀM SẠCH DỮ LIỆU ===")
    
    # Kiểm tra nếu chưa có file dữ liệu thô thì tạo file mẫu giả lập
    if not os.path.exists(input_path):
        print(f"Không tìm thấy '{input_path}'. Đang tạo dữ liệu mẫu để chạy thử...")
        sample_data = {
            "ID": [1, 2, 3, 4, None],
            "Timestamp": ["2026-10-01", "2026-10-02", None, "2026-10-04", "2026-10-05"],
            "Sensor_Value": [25.5, -999.0, 26.0, 24.8, 27.1], # -999 giả định là lỗi cảm biến
            "Status": ["OK", "ERROR", "OK", "OK", "WARNING"]
        }
        pd.DataFrame(sample_data).to_csv(input_path, index=False)
        print(f"Đã tạo xong file mẫu '{input_path}'.\n")

    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(input_path)
    print(f"Số dòng ban đầu: {len(df)}")

    # Làm sạch: Xóa dòng thiếu dữ liệu cốt lõi (ID hoặc Timestamp)
    df_clean = df.dropna(subset=['ID', 'Timestamp'])

    # Xử lý giá trị lỗi (ví dụ thay thế giá trị -999 thành trung bình của cột)
    valid_mean = df_clean[df_clean['Sensor_Value'] != -999.0]['Sensor_Value'].mean()
    df_clean.loc[df_clean['Sensor_Value'] == -999.0, 'Sensor_Value'] = valid_mean

    # Xuất ra file sạch mới
    df_clean.to_csv(output_path, index=False)
    print(f"Đã làm sạch và lưu tại: {output_path} (Còn lại {len(df_clean)} dòng)")

    # Thống kê cơ bản phục vụ EDA
    print("\n=== BẢNG THỐNG KÊ CƠ BẢN ===")
    print(df_clean.describe(include='all'))

if __name__ == "__main__":
    clean_and_analyze_csv(INPUT_FILE, OUTPUT_FILE)
