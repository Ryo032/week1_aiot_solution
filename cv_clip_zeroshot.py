"""
Kịch bản Thực hành Ngày 3 (Tuần 6): Computer Vision – Mô hình thị giác hiện đại (CLIP).
1. Cài đặt toán học lõi Zero-shot CLIP: Chuẩn hóa L2 -> Cosine Similarity -> Logit Scale -> Softmax.
2. Phân loại ảnh Zero-shot thực tế với mô hình tiền huấn luyện Hugging Face (openai/clip-vit-base-patch32).
"""

import logging
import os
from typing import Any

import numpy as np

# Tránh xung đột thư viện OpenMP trên Windows khi chạy chung PyTorch với NumPy
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("clip_zeroshot")

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def clip_zeroshot_classify(
    image_emb: np.ndarray,
    text_embs: np.ndarray,
    labels: list[str],
    logit_scale: float = 10.0,
) -> list[dict[str, Any]]:
    """
    Tính xác suất phân loại Zero-shot CLIP từ vector ảnh (d,) và ma trận vector nhãn (K, d).
    """
    img_norm = image_emb / (np.linalg.norm(image_emb) + 1e-12)
    txt_norm = text_embs / (np.linalg.norm(text_embs, axis=-1, keepdims=True) + 1e-12)

    # Tích vô hướng của các vector đã chuẩn hóa L2 chính là Cosine Similarity
    cosines = txt_norm @ img_norm

    # Phóng đại bằng hệ số nhiệt độ (1 / tau) trước khi đưa qua Softmax
    logits = cosines * logit_scale

    # Softmax ổn định số học
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)

    ranked_indices = np.argsort(-probs)
    results: list[dict[str, Any]] = []
    for idx in ranked_indices:
        i = int(idx)
        results.append(
            {
                "label": labels[i],
                "cosine_similarity": round(float(cosines[i]), 4),
                "probability": round(float(probs[i]), 4),
            }
        )
    return results


class ClipZeroShotClassifier:
    """Bộ phân loại ảnh Zero-shot sử dụng CLIP tiền huấn luyện từ Hugging Face."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL) -> None:
        from transformers import (  # type: ignore[import-not-found]
            CLIPModel,
            CLIPProcessor,
        )

        self.model_name = model_name
        logger.info("Dang nap CLIPProcessor va CLIPModel: %s", model_name)
        self.processor: Any = CLIPProcessor.from_pretrained(model_name)
        self.model: Any = CLIPModel.from_pretrained(model_name)
        self.model.eval()

    def predict(
        self,
        image: Any,
        candidate_labels: list[str],
        prompt_template: str = "a photo of {}",
    ) -> list[dict[str, Any]]:
        """Phân loại 1 ảnh theo danh sách nhãn ngôn ngữ tự nhiên bất kỳ (Zero-shot)."""
        import torch  # type: ignore[import-not-found]

        # Áp dụng Prompt Engineering bọc nhãn vào câu hoàn chỉnh
        prompts = [prompt_template.format(lbl) for lbl in candidate_labels]
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = torch.softmax(logits, dim=-1)

        scores = probs.cpu().numpy()
        ranked = np.argsort(-scores)
        return [
            {
                "label": candidate_labels[int(idx)],
                "prompt": prompts[int(idx)],
                "probability": round(float(scores[int(idx)]), 4),
            }
            for idx in ranked
        ]


if __name__ == "__main__":
    from PIL import Image

    sample_image: Any = None
    if os.path.isdir("images_raw"):
        for fname in sorted(os.listdir("images_raw")):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join("images_raw", fname)
                sample_image = Image.open(img_path).convert("RGB")
                logger.info("Su dung anh thuc te tu repo: %s", img_path)
                break

    if sample_image is None:
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :, 0] = 230
        arr[60:160, 60:160, 1] = 180
        sample_image = Image.fromarray(arr)
        logger.info("Tao anh mau 224x224 de chay thu nghiem CLIP Zero-shot.")

    labels_to_test = [
        "a sports car on the road",
        "a factory worker wearing a safety helmet",
        "fire and smoke hazard",
        "an industrial invoice document",
    ]
    clf = ClipZeroShotClassifier()
    preds = clf.predict(sample_image, labels_to_test)
    for item in preds:
        logger.info("[%.2f%%] %s", item["probability"] * 100, item["label"])
