"""
Kịch bản Thực hành Ngày 5 (Tuần 6): Công cụ – Triển khai Pilot & Quản lý Secrets.
1. Tầng nạp cấu hình an toàn: ưu tiên đọc bí mật từ tệp Docker Secrets (<ENV>_FILE),
   sau đó mới lùi về biến môi trường thông thường khi chạy cục bộ.
2. Bộ kiểm toán cấu hình tự động (Pilot Security Auditor): kiểm tra các nguyên tắc
   an toàn bắt buộc trong docker-compose.pilot.yml trước khi đưa lên máy chủ thử nghiệm.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pilot_config")

DEFAULT_COMPOSE_FILE = "docker-compose.pilot.yml"


class PilotSettings(BaseModel):
    """Cấu hình vận hành chuẩn hóa cho môi trường Pilot/Staging (12-Factor App)."""

    app_version: str
    pilot_domain: str
    log_level: str
    model_version: str
    api_secret_key: str


def load_secret(env_name: str, default: str = "") -> str:
    """
    Đọc thông tin bí mật theo thứ tự ưu tiên chuẩn Docker Secrets:
    1. Nếu biến <env_name>_FILE trỏ tới một tệp tồn tại (ví dụ /run/secrets/...),
       đọc nội dung tệp và cắt khoảng trắng thừa ở hai đầu.
    2. Ngược lại, lùi về đọc biến môi trường <env_name> hoặc giá trị mặc định.
    """
    file_path = os.getenv(f"{env_name}_FILE")
    if file_path:
        secret_file = Path(file_path)
        if secret_file.is_file():
            return secret_file.read_text(encoding="utf-8").strip()
        logger.warning("Khong tim thay tep secret tai duong dan: %s", file_path)
    return os.getenv(env_name, default)


def load_pilot_settings() -> PilotSettings:
    """Tổng hợp cấu hình thường và khóa bí mật thành đối tượng Pydantic đã xác thực."""
    return PilotSettings(
        app_version=os.getenv("APP_VERSION", "v1.2.4"),
        pilot_domain=os.getenv("PILOT_DOMAIN", "localhost"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        model_version=os.getenv("MODEL_VERSION", "v1.2.4"),
        api_secret_key=load_secret("AI_API_KEY", default="dev-local-key"),
    )


def audit_pilot_compose(compose_path: str | Path) -> list[str]:
    """
    Kiểm toán bảo mật và độ ổn định của tệp docker-compose.pilot.yml.
    Trả về danh sách lỗi vi phạm (danh sách rỗng nghĩa là đạt 100% chuẩn Pilot).
    """
    path = Path(compose_path)
    if not path.is_file():
        return [f"Khong tim thay tep cau hinh: {path}"]

    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return ["Noi dung YAML cua docker-compose khong hop le."]

    violations: list[str] = []
    services: dict[str, Any] = raw.get("services", {})
    networks: dict[str, Any] = raw.get("networks", {})

    # 1. Kiểm tra mạng nội bộ bắt buộc phải cô lập bằng internal: true
    internal_net = networks.get("internal_net")
    if not isinstance(internal_net, dict) or internal_net.get("internal") is not True:
        violations.append("Mang 'internal_net' chua bat co co lap 'internal: true'.")

    # 2. Kiểm tra dịch vụ lõi không được phơi cổng trần ra ngoài Host bằng 'ports'
    for backend_svc in ("ai-inference", "redis-cache"):
        svc_cfg = services.get(backend_svc)
        if not isinstance(svc_cfg, dict):
            violations.append(f"Thieu dich vu bat buoc '{backend_svc}'.")
            continue
        if "ports" in svc_cfg:
            violations.append(f"Dich vu noi bo '{backend_svc}' khong duoc dung 'ports' mo ra ngoai.")

    # 3. Kiểm tra Reverse Proxy chờ điều kiện service_healthy của dịch vụ AI
    proxy_cfg = services.get("caddy-proxy")
    if not isinstance(proxy_cfg, dict):
        violations.append("Thieu dich vu Reverse Proxy 'caddy-proxy'.")
    else:
        depends = proxy_cfg.get("depends_on", {})
        ai_dep = depends.get("ai-inference", {}) if isinstance(depends, dict) else {}
        if not isinstance(ai_dep, dict) or ai_dep.get("condition") != "service_healthy":
            violations.append("'caddy-proxy' phai cho 'ai-inference' o dieu kien 'service_healthy'.")

    # 4. Kiểm tra Redis có đặt giới hạn bộ nhớ và chính sách xóa khóa allkeys-lru
    redis_cfg = services.get("redis-cache")
    if isinstance(redis_cfg, dict):
        cmd = redis_cfg.get("command", [])
        cmd_text = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if "--maxmemory" not in cmd_text or "allkeys-lru" not in cmd_text:
            violations.append("'redis-cache' phai cau hinh '--maxmemory' va 'allkeys-lru'.")

    # 5. Kiểm tra dịch vụ AI có HEALTHCHECK và nạp Secrets an toàn
    ai_cfg = services.get("ai-inference")
    if isinstance(ai_cfg, dict):
        if "healthcheck" not in ai_cfg:
            violations.append("'ai-inference' thieu khai bao 'healthcheck'.")
        if not ai_cfg.get("secrets"):
            violations.append("'ai-inference' chua gan Docker 'secrets'.")

    # 6. Kiểm tra mọi Container đều có trần RAM, chính sách tự khởi động và Log Rotation
    for svc_name, svc_cfg in services.items():
        if not isinstance(svc_cfg, dict):
            continue
        if "mem_limit" not in svc_cfg:
            violations.append(f"Dich vu '{svc_name}' thieu gioi han RAM 'mem_limit'.")
        if "restart" not in svc_cfg:
            violations.append(f"Dich vu '{svc_name}' thieu chinh sach 'restart'.")
        log_opts = svc_cfg.get("logging", {}).get("options", {})
        if not isinstance(log_opts, dict) or "max-size" not in log_opts:
            violations.append(f"Dich vu '{svc_name}' thieu cau hinh xoay vong log 'max-size'.")

    return violations


if __name__ == "__main__":
    settings = load_pilot_settings()
    logger.info(
        "Cau hinh Pilot: domain=%s | version=%s | log_level=%s",
        settings.pilot_domain,
        settings.app_version,
        settings.log_level,
    )

    issues = audit_pilot_compose(DEFAULT_COMPOSE_FILE)
    if issues:
        for err in issues:
            logger.error("[VI PHAM PILOT] %s", err)
        raise SystemExit(1)

    logger.info(
        "[DAT CHUAN PILOT] '%s' vuot qua toan bo cac quy tac bao mat va on dinh.",
        DEFAULT_COMPOSE_FILE,
    )
