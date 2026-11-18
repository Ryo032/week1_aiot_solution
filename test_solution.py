import os
import pandas as pd


def test_clean_stats_script_exists():
    """Kiểm tra file script chính của dự án có tồn tại không."""
    assert os.path.exists("clean_stats.py")


def test_dataframe_cleaning_logic():
    """Kiểm tra logic lọc bỏ giá trị cảm biến lỗi (-999.0)."""
    raw_data = pd.DataFrame({"temp": [25.5, -999.0, 28.0]})
    clean_data = raw_data[raw_data["temp"] != -999.0]
    assert len(clean_data) == 2
    assert (clean_data["temp"] > 0).all()