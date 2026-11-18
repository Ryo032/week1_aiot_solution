import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

# ==============================================================================
# 1. THIẾT LẬP HỆ THỐNG LOGGING CHUẨN (THAY THẾ CHO HÀM PRINT() TRUYỀN THỐNG)
# ==============================================================================
# - level=logging.INFO (Mức 20): Chỉ cho phép các bản tin từ INFO (20), WARNING (30),
#   ERROR (40), CRITICAL (50) đi qua; tự động lọc bỏ các bản tin DEBUG (10) rác.
# - format: Tự động đính kèm Thời gian (asctime) + Cấp độ lỗi (levelname) + Tên module (name).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Khởi tạo đối tượng logger riêng cho gói thư viện aiot_pkg
logger = logging.getLogger("aiot_pkg")


# ==============================================================================
# 2. QUẢN LÝ CẤU HÌNH TÁCH BIỆT & KIỂM TRA KIỂU TĨNH (TYPE HINTS CHO MYPY)
# ==============================================================================
# @dataclass(frozen=True): Tạo lớp chứa thông số cấu hình ở chế độ Bất biến (Immutable).
# Khi frozen=True, mọi hành vi cố tình sửa đổi ngưỡng (ví dụ: config.temp_threshold = 999)
# trong lúc chương trình đang chạy đa luồng sẽ bị chặn đứng để tránh lỗi Race Condition!
@dataclass(frozen=True)
class SensorConfig:
    """Khuôn mẫu lưu trữ cấu hình ngưỡng cảm biến (tách biệt khỏi logic code)."""

    temp_threshold: float = 35.0  # Ngưỡng nhiệt độ cảnh báo mặc định (Độ C)
    error_code: float = -999.0  # Mã lỗi phần cứng mặc định của cảm biến


# Chú thích kiểu (Type Hints) tường minh:
# - Đầu vào bắt buộc: value kiểu 'float', config kiểu 'SensorConfig'
# - Đầu ra cam kết: trả về chuỗi 'str' ("OK" hoặc "WARNING")
# Công cụ 'mypy' sẽ đọc chữ ký này để bắt lỗi ngay nếu ai đó truyền chuỗi "35.0" vào value.
def validate_reading(value: float, config: SensorConfig) -> str:
    """Kiểm tra giá trị cảm biến và ghi nhật ký theo đúng phân cấp Logging."""
    if value == config.error_code:
        # Ghi log cấp WARNING (30) trước khi ném ngoại lệ
        logger.warning("Phat hien ma loi phan cung cam bien: %s", value)
        raise ValueError("Gia tri cam bien bi loi (-999.0)")

    if value > config.temp_threshold:
        # Ghi log cấp INFO (20) khi phát hiện nhiệt độ vượt ngưỡng
        logger.info(
            "Canh bao nhiet do cao: %.2f > %.2f",
            value,
            config.temp_threshold,
        )
        return "WARNING"

    return "OK"


# ==============================================================================
# 3. XỬ LÝ TÁC VỤ I/O-BOUND BẰNG BẤT ĐỒNG BỘ (ASYNCIO - SINGLE THREAD EVENT LOOP)
# ==============================================================================
# 'async def' định nghĩa một Coroutine (hàm bất đồng bộ siêu nhẹ, chỉ tốn vài KB RAM).
async def _simulate_webhook_io(station_id: int, delay: float = 0.1) -> str:
    """Mô phỏng tác vụ I/O-bound (gửi cảnh báo qua mạng mất 'delay' giây chờ phản hồi)."""
    # LƯU Ý QUAN TRỌNG: Phải dùng 'await asyncio.sleep(delay)' thay vì 'time.sleep(delay)'.
    # Khi gặp từ khóa 'await', hàm này chủ động tạm dừng và nhường ngay CPU về cho Event Loop
    # để Event Loop bắn tiếp gói tin của các trạm tiếp theo mà không phải ngồi chờ!
    await asyncio.sleep(delay)
    return f"ACK_STATION_{station_id}"


async def run_async_webhooks(num_stations: int, delay: float = 0.1) -> list[str]:
    """Kích hoạt đồng thời N tác vụ chờ mạng trên 1 luồng duy nhất bằng asyncio.gather."""
    # Tạo danh sách N Coroutines (chưa chạy ngay mà chỉ mới xếp lịch)
    tasks = [_simulate_webhook_io(i, delay) for i in range(num_stations)]
    # asyncio.gather(*tasks): Đẩy toàn bộ N tác vụ vào Event Loop chạy đồng thời cùng lúc.
    # Tổng thời gian hoàn thành N trạm sẽ chỉ xấp xỉ thời gian chờ của 1 trạm (~delay giây)!
    results: list[str] = await asyncio.gather(*tasks)
    return results


# ==============================================================================
# 4. XỬ LÝ TÁC VỤ I/O-BOUND BẰNG ĐA LUỒNG (THREADING - SHARED MEMORY)
# ==============================================================================
def _blocking_io_task(station_id: int, delay: float = 0.1) -> str:
    """Mô phỏng thư viện I/O đồng bộ cũ (không hỗ trợ async/await, dùng time.sleep)."""
    # Trong lúc một OS Thread nằm chờ ở lệnh I/O hoặc time.sleep(), trình thông dịch CPython
    # sẽ tạm thời NHẢ khóa GIL của luồng đó ra, giúp các Thread khác vẫn chạy song song được.
    time.sleep(delay)
    return f"THREAD_ACK_{station_id}"


def run_threaded_io_tasks(
    num_stations: int,
    delay: float = 0.1,
    max_workers: int = 5,
) -> list[str]:
    """Dùng ThreadPoolExecutor (Đa luồng) cho các tác vụ I/O đồng bộ truyền thống."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Chia đều các tác vụ chờ I/O cho tối đa 'max_workers' luồng hệ điều hành
        futures = [executor.submit(_blocking_io_task, i, delay) for i in range(num_stations)]
        return [f.result() for f in futures]


# ==============================================================================
# 5. XỬ LÝ TÁC VỤ CPU-BOUND BẰNG ĐA TIẾN TRÌNH (MULTIPROCESSING - VƯỢT KHÓA GIL)
# ==============================================================================
def cpu_heavy_filter(n: int) -> int:
    """
    Mô phỏng tác vụ CPU-bound: Vòng lặp tính toán số học nặng thuần Python.
    Vì thao tác cộng/nhân liên tục trên đối tượng số nguyên PyObject, luồng chạy hàm này
    sẽ GIỮ CHẶT khóa GIL. Nếu dùng 'threading', các luồng sẽ khóa chân nhau!
    """
    total = 0
    for i in range(n):
        total += (i * i) % 97
    return total


def run_parallel_cpu_tasks(workloads: list[int], max_workers: int = 2) -> list[int]:
    """
    Sử dụng ProcessPoolExecutor (multiprocessing) để tạo các Tiến trình (Processes) độc lập.
    Mỗi tiến trình con sở hữu 1 trình thông dịch Python riêng và 1 khóa GIL riêng biệt,
    giúp tính toán song song thực sự (True Parallelism) trên nhiều nhân CPU vật lý.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Lưu ý: Dữ liệu truyền vào 'workloads' sẽ được đóng gói qua cơ chế IPC Pickle
        results = list(executor.map(cpu_heavy_filter, workloads))
    return results


# ==============================================================================
# KHỐI BẢO VỆ BẮT BUỘC TRÊN WINDOWS KHI DÙNG MULTIPROCESSING
# ==============================================================================
# Trên Windows, multiprocessing dùng cơ chế 'spawn' (tiến trình con tự import lại file này).
# Nếu không có 'if __name__ == "__main__":', tiến trình con sẽ đẻ đệ quy vô tận (Fork Bomb)!
if __name__ == "__main__":
    cfg = SensorConfig()
    logger.info("Ket qua kiem tra 38.5C: %s", validate_reading(38.5, cfg))

    # Đo thời gian chạy 10 tác vụ I/O-bound bằng asyncio
    t0 = time.perf_counter()
    acks = asyncio.run(run_async_webhooks(10, delay=0.1))
    logger.info(
        "Asyncio hoan thanh %d webhooks trong %.3fs (Tuan tu mat 1.000s)",
        len(acks),
        time.perf_counter() - t0,
    )

    # Đo thời gian chạy tác vụ CPU-bound bằng multiprocessing
    t1 = time.perf_counter()
    cpu_res = run_parallel_cpu_tasks([500_000, 500_000], max_workers=2)
    logger.info(
        "Multiprocessing hoan thanh 2 tac vu CPU-bound trong %.3fs: %s",
        time.perf_counter() - t1,
        cpu_res,
    )
