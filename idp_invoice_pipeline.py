"""
Kịch bản Thực hành Ngày 5 (Tuần 5): Process Automation – IDP (Intelligent Document Processing).
Luồng xử lý trọn gói:
1. Tính toán chỉ số CER (Character Error Rate) bằng khoảng cách Levenshtein.
2. Sinh ảnh hóa đơn giả lập (1 hóa đơn chuẩn STP, 1 hóa đơn lệch tiền kích hoạt HITL).
3. Chạy OCR sâu (PaddleOCR chạy trên ONNX qua rapidocr-onnxruntime) lấy text, tọa độ và confidence.
4. Trích xuất dữ liệu vào Pydantic JSON Schema, kiểm tra chéo toán học & định tuyến STP / HITL.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("idp_invoice_pipeline")


class InvoiceData(BaseModel):
    """Schema JSON chuẩn hóa cho hóa đơn VAT (sẵn sàng nạp vào ERP/Database)."""

    invoice_number: str = Field(..., description="Mã số hóa đơn")
    tax_id: str = Field(..., description="Mã số thuế doanh nghiệp (10 hoặc 13 chữ số)")
    invoice_date: str = Field(..., description="Ngày lập hóa đơn (YYYY-MM-DD)")
    subtotal: float = Field(..., ge=0.0, description="Tổng tiền hàng trước thuế (VNĐ)")
    vat_amount: float = Field(..., ge=0.0, description="Tiền thuế GTGT (VNĐ)")
    total_amount: float = Field(..., ge=0.0, description="Tổng tiền thanh toán (VNĐ)")


class IDPResult(BaseModel):
    """Kết quả đầu ra của pipeline IDP kèm trạng thái định tuyến STP hoặc HITL."""

    source_file: str
    invoice: InvoiceData
    min_confidence: float
    avg_confidence: float
    math_valid: bool
    tax_id_valid: bool
    routing_decision: str  # "STP_AUTO_APPROVE" hoặc "HITL_HUMAN_REVIEW"
    review_reasons: list[str]


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Tính tỷ lệ lỗi ký tự CER = (S + D + I) / N bằng quy hoạch động (Levenshtein Distance).
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0

    n_rows = len(reference) + 1
    n_cols = len(hypothesis) + 1
    dp = [[0] * n_cols for _ in range(n_rows)]

    for i in range(n_rows):
        dp[i][0] = i
    for j in range(n_cols):
        dp[0][j] = j

    for i in range(1, n_rows):
        for j in range(1, n_cols):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    edit_distance = dp[n_rows - 1][n_cols - 1]
    return float(edit_distance / len(reference))


def generate_sample_invoices(output_dir: str = "idp_samples") -> list[Path]:
    """
    Sinh 2 ảnh hóa đơn mẫu để kiểm thử luồng IDP:
    - invoice_stp.png: Rõ nét, MST đủ 10 số, Subtotal + VAT == Total -> Đạt chuẩn STP.
    - invoice_hitl.png: Bị lệch tổng tiền (mô phỏng OCR đọc nhầm 8 thành 3) -> Bị chặn sang HITL.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    samples = [
        (
            "invoice_stp.png",
            [
                "VAT INVOICE",
                "Invoice No: HD-2026-001",
                "Tax ID: 0312845679",
                "Date: 2026-11-21",
                "Item 1: AIoT Gateway Module - 6000000",
                "Item 2: Industrial Camera - 4000000",
                "Subtotal: 10000000",
                "VAT: 800000",
                "Total: 10800000",
            ],
        ),
        (
            "invoice_hitl.png",
            [
                "VAT INVOICE",
                "Invoice No: HD-2026-003",
                "Tax ID: 0312845679",
                "Date: 2026-11-21",
                "Item 1: Edge Server Node - 8000000",
                "Subtotal: 8000000",
                "VAT: 640000",
                "Total: 3640000",  # Lệch 5,000,000 VNĐ so với 8,640,000
            ],
        ),
    ]

    created_files: list[Path] = []
    for filename, lines in samples:
        img = np.full((520, 760, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (740, 500), (60, 60, 60), 2)

        y_cursor = 70
        for idx, text in enumerate(lines):
            scale = 1.0 if idx == 0 else 0.72
            thickness = 2
            cv2.putText(
                img,
                text,
                (50, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (15, 15, 15),
                thickness,
                cv2.LINE_AA,
            )
            y_cursor += 50

        file_path = out_path / filename
        cv2.imwrite(str(file_path), img)
        created_files.append(file_path)

    logger.info("Da sinh %d anh hoa don mau tai: %s", len(created_files), out_path)
    return created_files


def run_ocr_on_image(image_path: Path) -> list[dict[str, Any]]:
    """
    Chạy OCR sâu (PaddleOCR ONNX) trên ảnh hóa đơn và sắp xếp dòng chữ theo tọa độ đọc (y, x).
    """
    from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]

    ocr_engine = RapidOCR()
    raw_result, _ = ocr_engine(str(image_path))
    if not raw_result:
        return []

    parsed_blocks: list[dict[str, Any]] = []
    for box, text, score in raw_result:
        y_top = float(box[0][1])
        x_left = float(box[0][0])
        parsed_blocks.append(
            {
                "text": str(text).strip(),
                "confidence": float(score),
                "x": x_left,
                "y": y_top,
            }
        )

    # Sắp xếp từ trên xuống dưới, trái sang phải (Layout Sorting)
    parsed_blocks.sort(key=lambda item: (round(item["y"] / 20.0), item["x"]))
    return parsed_blocks


def extract_invoice_schema(ocr_blocks: list[dict[str, Any]]) -> InvoiceData:
    """
    Chuẩn hóa văn bản OCR thành đối tượng Pydantic InvoiceData.
    (Trong môi trường Production có API Key, hàm này gọi LLM Structured Output với cùng Schema).
    """
    full_text = "\n".join(block["text"] for block in ocr_blocks)

    def _find_str(pattern: str, default: str = "UNKNOWN") -> str:
        match = re.search(pattern, full_text, re.IGNORECASE)
        return match.group(1).strip() if match else default

    def _find_amount(pattern: str) -> float:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if not match:
            return 0.0
        digits = re.sub(r"[^\d.]", "", match.group(1))
        return float(digits) if digits else 0.0

    return InvoiceData(
        invoice_number=_find_str(r"Invoice\s*No[:\s]+([A-Z0-9-]+)"),
        tax_id=re.sub(r"\D", "", _find_str(r"Tax\s*ID[:\s]+([0-9\s-]+)", "")),
        invoice_date=_find_str(r"Date[:\s]+(\d{4}-\d{2}-\d{2})"),
        subtotal=_find_amount(r"Subtotal[:\s]+([0-9.,]+)"),
        vat_amount=_find_amount(r"VAT[:\s]+([0-9.,]+)"),
        total_amount=_find_amount(r"\bTotal[:\s]+([0-9.,]+)"),
    )


def evaluate_and_route_idp(
    source_file: str,
    invoice: InvoiceData,
    confidences: list[float],
    conf_threshold: float = 0.85,
    math_tolerance: float = 1.0,
) -> IDPResult:
    """
    Bộ lọc 2 tầng (Automated Validation + Confidence Threshold) quyết định STP hay HITL.
    """
    min_conf = min(confidences) if confidences else 0.0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # Tầng 1: Kiểm tra ràng buộc cứng (Toán học & Định dạng MST)
    expected_total = invoice.subtotal + invoice.vat_amount
    math_valid = abs(expected_total - invoice.total_amount) <= math_tolerance
    tax_id_valid = len(invoice.tax_id) in (10, 13) and invoice.tax_id.isdigit()

    reasons: list[str] = []
    if not math_valid:
        reasons.append(
            f"Lech toan hoc: Subtotal ({invoice.subtotal:.0f}) + VAT ({invoice.vat_amount:.0f}) "
            f"!= Total ({invoice.total_amount:.0f})"
        )
    if not tax_id_valid:
        reasons.append(f"Ma so thue khong hop le: '{invoice.tax_id}'")

    # Tầng 2: Kiểm tra ngưỡng tin cậy tối thiểu
    if min_conf < conf_threshold:
        reasons.append(f"Do tin cay OCR thap ({min_conf:.3f} < {conf_threshold:.2f})")

    decision = "STP_AUTO_APPROVE" if not reasons else "HITL_HUMAN_REVIEW"

    return IDPResult(
        source_file=source_file,
        invoice=invoice,
        min_confidence=round(min_conf, 4),
        avg_confidence=round(avg_conf, 4),
        math_valid=math_valid,
        tax_id_valid=tax_id_valid,
        routing_decision=decision,
        review_reasons=reasons,
    )


def run_idp_pipeline(
    sample_dir: str = "idp_samples",
    report_dir: str = "reports",
) -> list[IDPResult]:
    """Chạy trọn gói pipeline IDP từ ảnh hóa đơn sang báo cáo JSON có định tuyến STP/HITL."""
    image_paths = generate_sample_invoices(sample_dir)
    results: list[IDPResult] = []

    for img_path in image_paths:
        blocks = run_ocr_on_image(img_path)
        invoice = extract_invoice_schema(blocks)
        confidences = [float(b["confidence"]) for b in blocks]
        idp_res = evaluate_and_route_idp(img_path.name, invoice, confidences)
        results.append(idp_res)
        logger.info(
            "[%s] Quyet dinh: %s | MinConf=%.3f | MathValid=%s",
            img_path.name,
            idp_res.routing_decision,
            idp_res.min_confidence,
            idp_res.math_valid,
        )

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "idp_invoice_results.json"
    report_file.write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Da xuat bao cao JSON IDP tai: %s", report_file)
    return results


if __name__ == "__main__":
    run_idp_pipeline()
