import asyncio
import os
from pathlib import Path

import pandas as pd
import pytest

# Import trực tiếp từ gói thư viện 'aiot_pkg' (đã được đăng ký vào môi trường ảo
# thông qua lệnh 'pip install -e .' mà không cần dùng thủ thuật sys.path.append)
from aiot_pkg import (
    SensorConfig,
    cpu_heavy_filter,
    run_async_webhooks,
    run_threaded_io_tasks,
    validate_reading,
)
from cv_yolo_custom import compute_iou, generate_custom_yolo_dataset
from model_evaluation_report import run_full_evaluation_pipeline


# Lưu ý: Vì trong pyproject.toml đã bật cờ 'disallow_untyped_defs = true' của mypy,
# nên ngay cả các hàm kiểm thử test_* cũng bắt buộc phải ghi rõ kiểu trả về '-> None'!
def test_clean_stats_script_exists() -> None:
    """1. Kiểm tra file script chính từ Tuần 1 vẫn tồn tại nguyên vẹn."""
    assert os.path.exists("clean_stats.py")


def test_dataframe_cleaning_logic() -> None:
    """2. Kiểm tra logic lọc bỏ giá trị cảm biến lỗi (-999.0) trên Pandas DataFrame."""
    raw_data = pd.DataFrame({"temp": [25.5, -999.0, 28.0]})
    clean_data = raw_data[raw_data["temp"] != -999.0]
    assert len(clean_data) == 2
    assert (clean_data["temp"] > 0).all()


def test_aiot_pkg_validate_reading() -> None:
    """3. Kiểm thử hàm validate_reading và lớp cấu hình bất biến SensorConfig."""
    cfg = SensorConfig(temp_threshold=30.0)
    assert validate_reading(25.0, cfg) == "OK"
    assert validate_reading(35.5, cfg) == "WARNING"

    # Kiểm tra xem hàm có ném đúng ngoại lệ ValueError khi gặp mã lỗi -999.0 không
    with pytest.raises(ValueError):
        validate_reading(-999.0, cfg)


def test_aiot_pkg_concurrency_models() -> None:
    """4. Kiểm thử cả 3 mô hình: Asyncio (I/O), Threading (I/O) và hàm tính toán CPU."""
    # Kiểm thử chạy đồng thời 5 tác vụ mạng bất đồng bộ bằng asyncio
    async_results = asyncio.run(run_async_webhooks(5, delay=0.02))
    assert len(async_results) == 5
    assert async_results[0] == "ACK_STATION_0"

    # Kiểm thử chạy đồng thời 4 tác vụ I/O đồng bộ bằng ThreadPoolExecutor
    thread_results = run_threaded_io_tasks(4, delay=0.02, max_workers=2)
    assert len(thread_results) == 4
    assert thread_results[0] == "THREAD_ACK_0"

    # Kiểm thử hàm tính toán nặng CPU-bound
    assert cpu_heavy_filter(100) > 0


def test_model_evaluation_report_pipeline(tmp_path: Path) -> None:
    """5. Kiểm thử đường ống xuất báo cáo đánh giá ML (ROC/PR, Calibration, SHAP, MLflow)."""
    report = run_full_evaluation_pipeline(output_dir=str(tmp_path))
    metrics = report["metrics"]

    # Kiểm tra các chỉ số đánh giá nằm trong miền hợp lệ và Brier Score cải thiện sau hiệu chuẩn
    assert 0.70 < metrics["roc_auc"] <= 1.0
    assert 0.40 < metrics["pr_auc"] <= 1.0
    assert metrics["brier_score_after_calibration"] <= metrics["brier_score_before_calibration"]

    # Kiểm tra các tệp hiện vật báo cáo (Artifacts) đã được xuất ra ổ cứng
    assert (tmp_path / "roc_pr_calibration.png").exists()
    assert (tmp_path / "shap_feature_importance.png").exists()
    assert (tmp_path / "evaluation_summary.json").exists()


def test_cv_iou_and_yolo_dataset_format(tmp_path: Path) -> None:
    """6. Kiểm thử công thức IoU và cấu trúc dữ liệu tự gán nhãn chuẩn YOLO."""
    # Kiểm chứng khớp đúng bài toán tính tay ở Tầng 3: IoU = 6400 / 13600 ≈ 0.4706
    iou_val = compute_iou((100, 100, 200, 200), (120, 120, 220, 220))
    assert abs(iou_val - (6400 / 13600)) < 1e-4

    # Kiểm chứng bộ sinh dữ liệu YOLO tạo đủ ảnh, nhãn chuẩn hóa (0..1) và dataset.yaml
    yaml_path = generate_custom_yolo_dataset(base_dir=str(tmp_path), n_train=4, n_val=2)
    assert yaml_path.exists()
    sample_lbl = tmp_path / "labels" / "train" / "sample_000.txt"
    assert sample_lbl.exists()
    first_line_parts = sample_lbl.read_text(encoding="utf-8").splitlines()[0].split()
    assert len(first_line_parts) == 5
    # Đảm bảo tọa độ đã chuẩn hóa về đoạn [0, 1]
    coords = [float(v) for v in first_line_parts[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)
