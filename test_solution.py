import asyncio
import os

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
