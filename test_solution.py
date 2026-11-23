import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

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
from idp_invoice_pipeline import (
    InvoiceData,
    compute_cer,
    evaluate_and_route_idp,
    run_idp_pipeline,
)
from inference_service import app as inference_app
from model_evaluation_report import run_full_evaluation_pipeline
from transformer_text_classifier import scaled_dot_product_attention


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


def test_idp_cer_and_stp_hitl_routing(tmp_path: Path) -> None:
    """7. Kiểm thử công thức CER, định tuyến STP/HITL và pipeline IDP đầu-cuối."""
    # Kiểm chứng khớp đúng bài toán tính tay ở Tầng 3: CER = 2 / 10 = 0.20
    assert abs(compute_cer("0312845679", "031234569") - 0.20) < 1e-6

    # Kiểm chứng trường hợp HD-003: Confidence cao (0.91) nhưng lệch tiền -> Vẫn phải chặn sang HITL
    bad_invoice = InvoiceData(
        invoice_number="HD-003",
        tax_id="0312845679",
        invoice_date="2026-11-21",
        subtotal=8_000_000.0,
        vat_amount=640_000.0,
        total_amount=3_640_000.0,
    )
    routed = evaluate_and_route_idp("hd_003.png", bad_invoice, [0.95, 0.91])
    assert routed.math_valid is False
    assert routed.routing_decision == "HITL_HUMAN_REVIEW"

    # Chạy pipeline thực tế trên 2 ảnh hóa đơn sinh tự động
    results = run_idp_pipeline(
        sample_dir=str(tmp_path / "samples"),
        report_dir=str(tmp_path / "reports"),
    )
    assert len(results) == 2
    assert results[0].routing_decision == "STP_AUTO_APPROVE"
    assert results[1].routing_decision == "HITL_HUMAN_REVIEW"
    assert (tmp_path / "reports" / "idp_invoice_results.json").exists()


def test_inference_service_versioning_cache_and_validation() -> None:
    """8. Kiểm thử dịch vụ suy luận: Versioning, Dynamic Batching, Cache-Aside & Validation."""
    with TestClient(inference_app) as client:
        # 1. Kiểm tra /health nạp đủ 2 phiên bản
        health = client.get("/health")
        assert health.status_code == 200
        assert set(health.json()["loaded_versions"]) == {"v1.0.0", "v2.0.0"}

        # 2. Gọi lần 1 (Cache Miss) với X-Request-ID và phiên bản v2.0.0
        headers = {"X-Request-ID": "trace-week6-001", "X-Model-Version": "v2.0.0"}
        res1 = client.post(
            "/v1/predict",
            json={"temperature": 45.0, "vibration": 10.0},
            headers=headers,
        )
        assert res1.status_code == 200
        assert res1.headers["X-Request-ID"] == "trace-week6-001"
        body1 = res1.json()
        assert body1["model_version"] == "v2.0.0"
        assert body1["cache_hit"] is False
        assert body1["label"] == "CRITICAL_FAULT"

        # 3. Gọi lần 2 cùng đầu vào & phiên bản -> Phải trúng Cache (Cache Hit)
        res2 = client.post(
            "/v1/predict",
            json={"temperature": 45.0, "vibration": 10.0},
            headers=headers,
        )
        assert res2.status_code == 200
        assert res2.json()["cache_hit"] is True

        # 4. Đổi sang v1.0.0 cùng đầu vào -> Phải Cache Miss (không bị Cache Pollution)
        res_v1 = client.post(
            "/v1/predict",
            json={"temperature": 45.0, "vibration": 10.0},
            headers={"X-Model-Version": "v1.0.0"},
        )
        assert res_v1.status_code == 200
        assert res_v1.json()["model_version"] == "v1.0.0"
        assert res_v1.json()["cache_hit"] is False

        # 5. Kiểm tra lỗi 404 (sai phiên bản) và 422 (nhiệt độ vượt biên vật lý)
        res_404 = client.post(
            "/v1/predict",
            json={"temperature": 30.0, "vibration": 5.0},
            headers={"X-Model-Version": "v9.9.9"},
        )
        assert res_404.status_code == 404

        res_422 = client.post(
            "/v1/predict",
            json={"temperature": 999.0, "vibration": 5.0},
        )
        assert res_422.status_code == 422


def test_transformer_scaled_dot_product_attention() -> None:
    """9. Kiểm thử công thức Scaled Dot-Product Self-Attention & Causal Masking (khớp Tầng 3)."""
    q = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 2.0]])
    k = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    v = np.array([[10.0, 0.0], [0.0, 20.0]])

    # 1. Kiểm tra Bidirectional Attention khớp đúng bài toán tính tay ở Tầng 3
    output, weights = scaled_dot_product_attention(q, k, v, causal_mask=False)
    assert output.shape == (2, 2)
    assert np.allclose(weights.sum(axis=-1), 1.0)
    assert abs(output[0, 0] - 7.3106) < 1e-2
    assert abs(output[0, 1] - 5.3788) < 1e-2

    # 2. Kiểm tra Causal Masking (Decoder LLM): từ đầu tiên không được nhìn từ thứ 2
    _, causal_weights = scaled_dot_product_attention(q, k, v, causal_mask=True)
    assert causal_weights[0, 1] < 1e-6
    assert abs(causal_weights[0, 0] - 1.0) < 1e-6
