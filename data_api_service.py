import time
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base, sessionmaker

# ==========================================
# PHẦN 1: PYTHON XỬ LÝ DỮ LIỆU NÂNG CAO (TUẦN 3)
# ==========================================

# 1. Decorator đo hiệu năng tốc độ chạy hàm
def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[HIỆU NĂNG] Hàm '{func.__name__}' chạy mất: {(end - start)*1000:.2f} ms")
        return result
    return wrapper

# 2. Generator đọc từng dòng dữ liệu để tiết kiệm RAM
def sensor_stream_generator(n_rows: int):
    for i in range(n_rows):
        yield {"ID": i, "Camera_ID": f"CAM_0{(i % 2) + 1}", "Sensor_Value": 25.0 + (i % 10)}

# 3. Gắn Decorator lên đầu hàm để tự động bấm giờ
@measure_time
def run_advanced_data_pipeline():
    # Đọc 10.000 dòng dữ liệu từ Generator
    df = pd.DataFrame(list(sensor_stream_generator(10000)))
    
    # NumPy Vector hóa: Đổi toàn bộ cột độ C sang độ F không dùng vòng lặp for
    df["Temp_F"] = np.round(df["Sensor_Value"].to_numpy() * 1.8 + 32, 2)
    
    # Pandas: merge (ghép bảng vị trí xưởng) + groupby
    zones = pd.DataFrame({"Camera_ID": ["CAM_01", "CAM_02"], "Zone": ["Xưởng A", "Xưởng B"]})
    df_merged = pd.merge(df, zones, on="Camera_ID")
    
    # Gom nhóm tính nhiệt độ trung bình theo từng Xưởng
    summary = df_merged.groupby("Zone")["Sensor_Value"].mean().to_dict()
    print("Kết quả gom nhóm (Groupby):", summary)
    return summary

# ==========================================
# PHẦN 2: DỊCH VỤ WEB FASTAPI & SQLALCHEMY (TUẦN 4)
# ==========================================

# Khởi tạo Database SQLite bằng SQLAlchemy ORM
engine = create_engine("sqlite:///./aiot_alerts.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class AlertDBModel(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String)
    sensor_value = Column(Float)
    status = Column(String)

Base.metadata.create_all(bind=engine)

# Khuôn mẫu Pydantic kiểm tra dữ liệu đầu vào
class SensorInput(BaseModel):
    camera_id: str
    sensor_value: float

app = FastAPI(title="AIoT Solution Engineer API - Tuần 3 & 4")

@app.get("/pipeline-stats")
async def get_pipeline_stats():
    stats = run_advanced_data_pipeline()
    return {"message": "Đã chạy pipeline tối ưu!", "zone_average_temp": stats}

@app.post("/predict-and-save")
async def predict_and_save(data: SensorInput):
    status = "WARNING" if data.sensor_value > 30.0 else "OK"
    db = SessionLocal()
    new_record = AlertDBModel(camera_id=data.camera_id, sensor_value=data.sensor_value, status=status)
    db.add(new_record)
    db.commit()
    db.close()
    return {"camera": data.camera_id, "temp": data.sensor_value, "ai_prediction": status, "saved_to_db": True}