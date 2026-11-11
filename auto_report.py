import pandas as pd
from datetime import datetime
import os

# Tự động lấy đường dẫn tuyệt đối của thư mục chứa file auto_report.py (Ổ đĩa G:\IOE\...)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_daily_report():
    today = datetime.now().strftime('%d/%m/%Y %H:%M')
    print(f"=== BÁO CÁO TỰ ĐỘNG CAMERA AIoT ({today}) ===")
    
    # 1. Đọc file CSV bằng đường dẫn tuyệt đối
    csv_path = os.path.join(PROJECT_DIR, 'data_clean.csv')
    df = pd.read_csv(csv_path)
    
    # 2. Tổng hợp số liệu
    total_records = len(df)
    avg_sensor = df['Sensor_Value'].mean()
    status_counts = df['Status'].value_counts().to_dict()
    
    # 3. Xuất nội dung báo cáo
    report_text = (
        f"BÁO CÁO NGÀY: {today}\n"
        f"- Tổng số bản ghi hợp lệ: {total_records}\n"
        f"- Giá trị cảm biến trung bình: {avg_sensor:.2f}\n"
        f"- Thống kê trạng thái (Status): {status_counts}\n"
        f"=> Trạng thái: Đã tổng hợp và sẵn sàng gửi Email cho Quản đốc.\n"
    )
    print(report_text)
    
    # 4. Ghi file txt bằng đường dẫn tuyệt đối
    log_path = os.path.join(PROJECT_DIR, 'daily_report_log.txt')
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(report_text + "-"*40 + "\n")

if __name__ == "__main__":
    generate_daily_report()