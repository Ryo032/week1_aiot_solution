"""
Kịch bản Thực hành Ngày 3 (Tuần 5): Báo cáo đánh giá đầy đủ cho một mô hình Học máy.
Bao gồm:
1. Xử lý dữ liệu mất cân bằng (class_weight='balanced' & Threshold Tuning).
2. Đánh giá song song đường cong ROC-AUC và PR-AUC (Precision-Recall).
3. Hiệu chuẩn xác suất (Probability Calibration - Sigmoid/Platt Scaling) & Brier Score.
4. Giải thích mô hình với SHAP (TreeExplainer) và Permutation Importance.
5. Theo dõi và lưu trữ thí nghiệm tự động với MLflow Tracking.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

# Sử dụng backend 'Agg' để vẽ và lưu biểu đồ ra file ảnh (.png) ổn định
# ngay cả trên môi trường máy chủ CI/CD (GitHub Actions) không có màn hình hiển thị.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ml_evaluator")

FEATURE_NAMES = [
    "temp_celsius",
    "vibration_mms",
    "pressure_bar",
    "current_amp",
    "humidity_pct",
    "noise_db",
]


def generate_imbalanced_sensor_data(
    n_samples: int = 2000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    1. Sinh tập dữ liệu cảm biến AIoT mất cân bằng (92% Bình thường : 8% Sự cố).
    Trả về DataFrame có tên cột rõ ràng để phục vụ giải thích SHAP.
    """
    x_raw, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=4,
        n_redundant=1,
        weights=[0.92, 0.08],
        flip_y=0.01,
        random_state=random_state,
    )
    x_df = pd.DataFrame(x_raw, columns=FEATURE_NAMES)
    return x_df, y


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """
    2. Tìm ngưỡng cắt quyết định tối ưu (tau*) giúp cực đại hóa chỉ số F1-Score
    trên đường cong Precision-Recall thay vì dùng ngưỡng cố định 0.50.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Tránh chia cho 0 khi cả precision và recall đều bằng 0
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / np.maximum(precisions[:-1] + recalls[:-1], 1e-8)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def run_full_evaluation_pipeline(output_dir: str = "reports") -> dict[str, Any]:
    """
    Thực thi toàn bộ đường ống huấn luyện, hiệu chuẩn, giải thích SHAP và ghi log MLflow.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # BƯỚC 1: CHUẨN BỊ DỮ LIỆU MẤT CÂN BẰNG (CHIA STRATIFIED TRAIN / TEST)
    # --------------------------------------------------------------------------
    x_df, y = generate_imbalanced_sensor_data(n_samples=2000, random_state=42)
    # Bắt buộc dùng stratify=y để giữ nguyên tỷ lệ 8% lỗi ở cả tập Train và tập Test
    x_train, x_test, y_train, y_test = train_test_split(x_df, y, test_size=0.30, stratify=y, random_state=42)

    # --------------------------------------------------------------------------
    # BƯỚC 2: HUẤN LUYỆN MÔ HÌNH CƠ SỞ & HIỆU CHUẨN XÁC SUẤT (PLATT SCALING)
    # --------------------------------------------------------------------------
    # Sử dụng class_weight="balanced" để phạt nặng khi đoán sót lớp thiểu số (Lớp 1)
    base_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    base_rf.fit(x_train, y_train)

    # Thực hiện Hiệu chuẩn xác suất (Probability Calibration) bằng phương pháp Sigmoid
    # thông qua Cross-Validation 5-fold ngay trên tập Train (tránh Data Leakage tập Test)
    calibrated_rf = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),
        method="sigmoid",
        cv=5,
    )
    calibrated_rf.fit(x_train, y_train)

    # Lấy xác suất dự đoán trên tập Test trước và sau khi hiệu chuẩn
    prob_uncalibrated = base_rf.predict_proba(x_test)[:, 1]
    prob_calibrated = calibrated_rf.predict_proba(x_test)[:, 1]

    # --------------------------------------------------------------------------
    # BƯỚC 3: TÍNH TOÁN ROC-AUC, PR-AUC, BRIER SCORE & TỐI ƯU NGƯỠNG CẮT (TAU*)
    # --------------------------------------------------------------------------
    roc_auc = float(roc_auc_score(y_test, prob_calibrated))
    pr_auc = float(average_precision_score(y_test, prob_calibrated))
    brier_before = float(brier_score_loss(y_test, prob_uncalibrated))
    brier_after = float(brier_score_loss(y_test, prob_calibrated))

    # So sánh F1-score tại ngưỡng mặc định (0.50) và ngưỡng tối ưu (tau*)
    pred_default = (prob_calibrated >= 0.50).astype(int)
    f1_default = float(f1_score(y_test, pred_default))

    opt_threshold, f1_optimal = find_optimal_threshold(y_test, prob_calibrated)
    pred_optimal = (prob_calibrated >= opt_threshold).astype(int)
    precision_opt = float(precision_score(y_test, pred_optimal))
    recall_opt = float(recall_score(y_test, pred_optimal))

    # --------------------------------------------------------------------------
    # BƯỚC 4: GIẢI THÍCH MÔ HÌNH VỚI SHAP (TREESHAP) & PERMUTATION IMPORTANCE
    # --------------------------------------------------------------------------
    # Tính Permutation Importance trên tập Test
    perm_result = permutation_importance(
        base_rf, x_test, y_test, n_repeats=5, random_state=42, scoring="average_precision"
    )
    perm_importances = {col: float(val) for col, val in zip(FEATURE_NAMES, perm_result.importances_mean)}

    # Tính giá trị Shapley bằng TreeExplainer trên tập mẫu kiểm thử
    explainer = shap.TreeExplainer(base_rf)
    x_test_sample = x_test.iloc[:150]
    shap_values_raw = explainer.shap_values(x_test_sample)

    # Tương thích đa phiên bản SHAP (list cho mỗi lớp hoặc mảng 3 chiều)
    if isinstance(shap_values_raw, list):
        shap_vals_pos = np.asarray(shap_values_raw[1])
    elif getattr(shap_values_raw, "ndim", 0) == 3:
        shap_vals_pos = np.asarray(shap_values_raw[:, :, 1])
    else:
        shap_vals_pos = np.asarray(shap_values_raw)

    mean_abs_shap = {
        col: float(val) for col, val in zip(FEATURE_NAMES, np.mean(np.abs(shap_vals_pos), axis=0))
    }

    # --------------------------------------------------------------------------
    # BƯỚC 5: VẼ VÀ XUẤT CÁC BIỂU ĐỒ BÁO CÁO (ROC, PR, CALIBRATION, SHAP)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Biểu đồ 1: Đường cong ROC
    fpr, tpr, _ = roc_curve(y_test, prob_calibrated)
    axes[0].plot(fpr, tpr, label=f"Calibrated RF (AUC = {auc(fpr, tpr):.3f})", lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", label="Random Guess (0.500)")
    axes[0].set_title("1. ROC Curve")
    axes[0].set_xlabel("False Positive Rate (FPR)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Biểu đồ 2: Đường cong Precision-Recall (PR Curve)
    prec, rec, _ = precision_recall_curve(y_test, prob_calibrated)
    baseline_pr = float(np.mean(y_test))
    axes[1].plot(rec, prec, color="darkorange", label=f"PR Curve (AP = {pr_auc:.3f})", lw=2)
    axes[1].axhline(baseline_pr, color="k", linestyle="--", label=f"Baseline ({baseline_pr:.3f})")
    axes[1].set_title("2. Precision-Recall Curve (Imbalanced)")
    axes[1].set_xlabel("Recall (TPR)")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    # Biểu đồ 3: Đường cong Hiệu chuẩn Xác suất (Calibration / Reliability Diagram)
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, prob_uncalibrated, n_bins=8)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, prob_calibrated, n_bins=8)
    axes[2].plot(
        prob_pred_uncal,
        prob_true_uncal,
        "s-",
        label=f"Uncalibrated (Brier={brier_before:.4f})",
    )
    axes[2].plot(
        prob_pred_cal,
        prob_true_cal,
        "o-",
        label=f"Calibrated (Brier={brier_after:.4f})",
    )
    axes[2].plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    axes[2].set_title("3. Probability Calibration Curve")
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Fraction of Positives (Empirical)")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    curves_plot_path = out_path / "roc_pr_calibration.png"
    fig.savefig(curves_plot_path, dpi=150)
    plt.close(fig)

    # Biểu đồ 4: So sánh SHAP Importance vs Permutation Importance
    fig_shap, ax_shap = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(FEATURE_NAMES))
    shap_means = [mean_abs_shap[c] for c in FEATURE_NAMES]
    ax_shap.barh(y_pos, shap_means, color="steelblue")
    ax_shap.set_yticks(y_pos)
    ax_shap.set_yticklabels(FEATURE_NAMES)
    ax_shap.invert_yaxis()
    ax_shap.set_xlabel("Mean |SHAP Value| (Impact on Model Output)")
    ax_shap.set_title("Global Feature Importance via TreeSHAP")
    ax_shap.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    shap_plot_path = out_path / "shap_feature_importance.png"
    fig_shap.savefig(shap_plot_path, dpi=150)
    plt.close(fig_shap)

    # Lưu báo cáo tổng hợp ra tệp JSON
    report_data: dict[str, Any] = {
        "dataset_summary": {
            "total_test_samples": int(len(y_test)),
            "positive_ratio": float(np.mean(y_test)),
        },
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score_before_calibration": round(brier_before, 4),
            "brier_score_after_calibration": round(brier_after, 4),
            "f1_at_default_0.5": round(f1_default, 4),
            "optimal_threshold": round(opt_threshold, 4),
            "f1_at_optimal_threshold": round(f1_optimal, 4),
            "precision_at_optimal": round(precision_opt, 4),
            "recall_at_optimal": round(recall_opt, 4),
        },
        "explainability": {
            "mean_abs_shap": {k: round(v, 4) for k, v in mean_abs_shap.items()},
            "permutation_importance": {k: round(v, 4) for k, v in perm_importances.items()},
        },
    }
    report_json_path = out_path / "evaluation_summary.json"
    report_json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    # --------------------------------------------------------------------------
    # BƯỚC 6: GHI NHẬN THÍ NGHIỆM TỰ ĐỘNG VÀO MLFLOW TRACKING
    # --------------------------------------------------------------------------

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("AIoT_Predictive_Maintenance_Week5")

    with mlflow.start_run(run_name="RF_Balanced_Calibrated_SHAP"):
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 6,
                "class_weight": "balanced",
                "calibration_method": "sigmoid",
            }
        )
        mlflow.log_metrics(
            {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_before": brier_before,
                "brier_after": brier_after,
                "f1_default": f1_default,
                "f1_optimal": f1_optimal,
                "optimal_threshold": opt_threshold,
            }
        )
        mlflow.log_artifact(str(curves_plot_path))
        mlflow.log_artifact(str(shap_plot_path))
        mlflow.log_artifact(str(report_json_path))

    logger.info("Hoan tat xuat bao cao danh gia mo hinh tai thu muc: %s", out_path)
    logger.info(
        "ROC-AUC=%.4f | PR-AUC=%.4f | Brier(Before->After)=%.4f->%.4f | F1(Opt)=%.4f",
        roc_auc,
        pr_auc,
        brier_before,
        brier_after,
        f1_optimal,
    )
    return report_data


if __name__ == "__main__":
    run_full_evaluation_pipeline()
