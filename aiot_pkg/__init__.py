"""
Tệp __init__.py đóng 2 vai trò cốt lõi trong chuẩn đóng gói Python:
1. Báo cho trình thông dịch Python và công cụ setuptools biết thư mục 'aiot_pkg'
   là một Gói thư viện (Package) có thể cài đặt và import, chứ không phải thư mục thường.
2. Gom các hàm/lớp quan trọng từ file con (core.py) đưa ra mặt tiền của gói,
   giúp người dùng chỉ cần gõ 'from aiot_pkg import SensorConfig' ngắn gọn.
"""

# Import các thành phần từ module con core.py đưa ra cấp độ gói (Package-level)
from aiot_pkg.core import (
    SensorConfig,
    cpu_heavy_filter,
    run_async_webhooks,
    run_parallel_cpu_tasks,
    run_threaded_io_tasks,
    validate_reading,
)

# Danh sách __all__ khai báo tường minh những hàm/lớp nào được phép xuất ra ngoài (Public API)
# khi người dùng gọi 'from aiot_pkg import *' và giúp công cụ mypy/ruff không báo lỗi F401 (import thừa).
__all__ = [
    "SensorConfig",
    "cpu_heavy_filter",
    "run_async_webhooks",
    "run_parallel_cpu_tasks",
    "run_threaded_io_tasks",
    "validate_reading",
]
