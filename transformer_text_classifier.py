"""
Kịch bản Thực hành Ngày 2 (Tuần 6): Deep Learning – Transformer & Hugging Face.
Bao gồm 2 phần:
1. Cài đặt toán học Scaled Dot-Product Self-Attention (hỗ trợ cả Bidirectional & Causal Masking).
2. Sử dụng mô hình tiền huấn luyện Hugging Face (DistilBERT) phân loại văn bản theo lô (Batch).
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
logger = logging.getLogger("transformer_classifier")

DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cài đặt công thức Scaled Dot-Product Self-Attention:
    Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V
    """
    d_k = q.shape[-1]
    # Chia cho sqrt(d_k) để ổn định phương sai trước khi đưa qua Softmax
    scores = (q @ k.T) / np.sqrt(d_k)

    if causal_mask:
        seq_len = scores.shape[0]
        # Tạo mặt nạ tam giác trên (cấm từ đứng trước nhìn trộm từ đứng sau trong Decoder LLM)
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)

    # Tính Softmax ổn định số học (trừ giá trị lớn nhất mỗi hàng để chống tràn số exp)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = attn_weights @ v
    return output, attn_weights


class PretrainedTextClassifier:
    """Bộ phân loại văn bản dùng mô hình Encoder-Only tiền huấn luyện từ Hugging Face."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.model_name = model_name
        logger.info("Dang nap Tokenizer va Model tien huan luyen: %s", model_name)
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Chuyển mô hình sang chế độ đánh giá (tắt Dropout để kết quả ổn định)
        self.model.eval()

    def predict_batch(self, texts: list[str], max_length: int = 512) -> list[dict[str, Any]]:
        """Phân loại một lô văn bản, tự động padding, attention_mask và cắt ngắn ở max_length."""
        import torch  # type: ignore[import-not-found]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Tắt tính đạo hàm khi suy luận để tiết kiệm RAM và tăng tốc độ
        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)

        id2label: dict[int, str] = self.model.config.id2label
        results: list[dict[str, Any]] = []
        for idx, text in enumerate(texts):
            pred_id = int(torch.argmax(probs[idx]).item())
            confidence = float(probs[idx, pred_id].item())
            results.append(
                {
                    "text": text,
                    "label": id2label[pred_id],
                    "confidence": round(confidence, 4),
                }
            )
        return results


if __name__ == "__main__":
    sample_logs = [
        "Sensor calibration succeeded and all factory edge nodes are operating smoothly.",
        "Critical overheating detected on motor 3, immediate shutdown required due to severe fault.",
        "System dashboard updated successfully with zero latency issues.",
    ]
    clf = PretrainedTextClassifier()
    predictions = clf.predict_batch(sample_logs)
    for item in predictions:
        logger.info(
            "[%s | %.2f%%] %s",
            item["label"],
            item["confidence"] * 100,
            item["text"],
        )
