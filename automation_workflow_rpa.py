import time
import logging
import urllib.request
import json
from playwright.sync_api import sync_playwright

# 1. Cấu hình Logging chuẩn (Vừa in ra màn hình, vừa lưu vào file automation_bot.log)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("automation_bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ==========================================================
# PHẦN 1: WORKFLOW TÍCH HỢP WEBHOOK API + RETRY (TUẦN 3 & 4)
# ==========================================================
def send_webhook_with_retry(alert_data, max_retries=3):
    webhook_url = "https://httpbin.org/post"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "aiot-secret-key-2026"
    }

    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"[Webhook - Lần thử {attempt}/{max_retries}] Đang đẩy cảnh báo tức thời...")
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(alert_data).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                if response.status == 200:
                    logging.info("[Webhook SUCCESS] Đã bắn cảnh báo thành công qua Webhook API (Status 200)!")
                    return True
        except Exception as e:
            logging.warning(f"[Retry] Lỗi kết nối lần {attempt}: {e}. Đợi 1 giây trước khi thử lại...")
            time.sleep(1)

    logging.error("[Webhook FAILED] Đã thử tối đa số lần nhưng vẫn thất bại!")
    return False

# ==========================================================
# PHẦN 2: RPA BOT TỰ ĐỘNG ĐIỀN FORM WEB BẰNG PLAYWRIGHT (TUẦN 4)
# ==========================================================
def run_playwright_rpa_bot(camera_id, issue_detail):
    logging.info("[RPA Bot] Khởi động trình duyệt Chromium ngầm (Playwright)...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Giả lập trang web nội bộ Legacy (không có API, chỉ có ô nhập liệu HTML)
        legacy_html = """
        <html><body>
            <h2>HỆ THỐNG BÁO CÁO NỘI BỘ NHÀ MÁY (LEGACY WEB 2010)</h2>
            <input id="cam_input" type="text" placeholder="Nhập mã Camera" />
            <input id="issue_input" type="text" placeholder="Nhập nội dung lỗi" />
            <button id="submit_btn" onclick="document.getElementById('result').innerText = 'ĐÃ LƯU BÁO CÁO: ' + document.getElementById('cam_input').value + ' - ' + document.getElementById('issue_input').value">Gửi báo cáo</button>
            <p id="result"></p>
        </body></html>
        """
        page.set_content(legacy_html)

        # Bot tự động gõ phím vào ô input và click nút Gửi
        page.fill("#cam_input", camera_id)
        page.fill("#issue_input", issue_detail)
        page.click("#submit_btn")

        confirmation = page.inner_text("#result")
        logging.info(f"[RPA Bot SUCCESS] Kết quả trích xuất từ giao diện Web: '{confirmation}'")
        browser.close()

if __name__ == "__main__":
    print("=== CHẠY THỬ NGHIỆM WORKFLOW WEBHOOK (TUẦN 3) & RPA PLAYWRIGHT (TUẦN 4) ===")
    event = {"camera_id": "CAM-ZONE-A1", "temp": 45.2, "issue": "Quá nhiệt & Không đội mũ bảo hộ"}

    # Ngã rẽ BPMN Gateway (Hình thoi): Kiểm tra điều kiện nhiệt độ > 40 độ C
    if event["temp"] > 40.0:
        send_webhook_with_retry(event, max_retries=3)
        run_playwright_rpa_bot(event["camera_id"], event["issue"])