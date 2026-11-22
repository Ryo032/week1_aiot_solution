"""
Kịch bản Thực hành Ngày 4 (Tuần 5): Computer Vision – Object Detection với YOLO.
Bao gồm:
1. Cài đặt hàm toán học tính IoU (Intersection over Union) chuẩn hóa.
2. Tự tạo và gán nhãn bộ dữ liệu theo đúng cấu trúc thư mục & định dạng YOLO (.txt + dataset.yaml).
3. Huấn luyện (Fine-tune) mô hình YOLOv8n trên dữ liệu tự gán nhãn và xuất ảnh suy luận.
"""

import logging
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cv_yolo_custom")


def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """
    Tính chỉ số IoU (Intersection over Union) giữa hai khung bao (x1, y1, x2, y2).
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_right - x_left)
    inter_h = max(0.0, y_bottom - y_top)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def generate_custom_yolo_dataset(
    base_dir: str = "yolo_custom_data",
    n_train: int = 16,
    n_val: int = 4,
    img_size: int = 320,
    seed: int = 42,
) -> Path:
    """
    Tự động tạo bộ dữ liệu ảnh giám sát an toàn và tệp nhãn chuẩn YOLO:
    - Lớp 0 (helmet): Mũ bảo hộ (mô phỏng bằng khối tròn màu vàng).
    - Lớp 1 (no_helmet): Không đội mũ (mô phỏng bằng khối vuông màu đỏ).
    Mỗi dòng trong tệp .txt có cấu trúc: <class_id> <x_center> <y_center> <width> <height> (chuẩn hóa 0..1).
    """
    rng = np.random.default_rng(seed)
    root = Path(base_dir).resolve()

    splits = {"train": n_train, "val": n_val}
    for split_name, count in splits.items():
        img_dir = root / "images" / split_name
        lbl_dir = root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            # Tạo nền nhà xưởng màu xám nhạt kèm nhiễu nhẹ
            img = rng.integers(180, 215, size=(img_size, img_size, 3), dtype=np.uint8)
            label_lines: list[str] = []

            # Vẽ 2 vật thể ngẫu nhiên trên mỗi bức ảnh và tính tọa độ nhãn YOLO
            for class_id in [0, 1]:
                box_w = int(rng.integers(48, 80))
                box_h = int(rng.integers(48, 80))
                x1 = int(rng.integers(10, img_size - box_w - 10))
                y1 = int(rng.integers(10, img_size - box_h - 10))
                x2 = x1 + box_w
                y2 = y1 + box_h

                if class_id == 0:
                    # Lớp 0 (helmet): Vẽ hình tròn màu vàng (BGR: 0, 215, 255)
                    center = (x1 + box_w // 2, y1 + box_h // 2)
                    radius = min(box_w, box_h) // 2
                    cv2.circle(img, center, radius, (0, 215, 255), -1)
                    cv2.circle(img, center, radius, (0, 140, 180), 2)
                else:
                    # Lớp 1 (no_helmet): Vẽ hình vuông màu đỏ (BGR: 40, 40, 230)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 230), -1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 150), 2)

                # Chuyển đổi tọa độ pixel (x1, y1, x2, y2) sang định dạng chuẩn YOLO (0..1)
                x_center = ((x1 + x2) / 2.0) / img_size
                y_center = ((y1 + y2) / 2.0) / img_size
                w_norm = box_w / float(img_size)
                h_norm = box_h / float(img_size)
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            img_path = img_dir / f"sample_{idx:03d}.jpg"
            lbl_path = lbl_dir / f"sample_{idx:03d}.txt"
            cv2.imwrite(str(img_path), img)
            lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    # Tạo tệp cấu hình dataset.yaml chuẩn của Ultralytics YOLO
    yaml_config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "helmet", 1: "no_helmet"},
    }
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_config, sort_keys=False), encoding="utf-8")
    logger.info("Da khoi tao bo du lieu tu gan nhan chuan YOLO tai: %s", root)
    return yaml_path


def train_and_infer_yolo(
    dataset_yaml: Path,
    report_dir: str = "reports",
    epochs: int = 3,
) -> Path:
    """
    Huấn luyện (Fine-tune) mô hình YOLOv8n trên dữ liệu tự gán nhãn và xuất ảnh dự đoán mẫu.
    """
    from ultralytics import YOLO  # type: ignore[import-not-found]

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Bat dau tai mo hinh pretrained YOLOv8n va huan luyen %d epochs...", epochs)
    model = YOLO("yolov8n.pt")

    # Huấn luyện nhẹ trên CPU với imgsz=320 để chạy nhanh và tiết kiệm RAM
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=320,
        batch=4,
        device="cpu",
        project="runs_yolo",
        name="safety_custom",
        exist_ok=True,
        verbose=False,
    )

    # Chạy suy luận (Inference) trên 1 ảnh trong tập Validation và lưu kết quả vào reports/
    val_img = dataset_yaml.parent / "images" / "val" / "sample_000.jpg"
    results = model.predict(source=str(val_img), imgsz=320, conf=0.15, device="cpu", verbose=False)

    annotated_img = results[0].plot()  # type: ignore[index, union-attr]
    output_img_path = out_dir / "yolo_inference_sample.jpg"
    cv2.imwrite(str(output_img_path), annotated_img)
    logger.info("Da luu anh ket qua suy luan YOLO tai: %s", output_img_path)
    return output_img_path


if __name__ == "__main__":
    yaml_file = generate_custom_yolo_dataset()
    train_and_infer_yolo(yaml_file, epochs=3)
