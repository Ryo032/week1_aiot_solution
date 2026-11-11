import cv2
import os

INPUT_DIR = "images_raw"
OUTPUT_DIR = "images_contours"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_classical_cv():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"=== BẮT ĐẦU TÁCH ĐỐI TƯỢNG CHO {len(files)} ẢNH ===")

    for filename in files:
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 1. Chuẩn hóa kích thước 640x640 và chuyển sang ảnh xám
        img_resized = cv2.resize(img, (640, 640))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # 2. Lọc nhiễu (Gaussian Blur) giúp làm mịn bề mặt ảnh, tránh bắt nhầm hạt bụi
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Ngưỡng hóa (Thresholding - Otsu) tách hẳn ảnh thành 2 màu Đen & Trắng (Nền & Vật thể)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Tìm đường viền (Contours) và vẽ khung xanh lá quanh các vật thể lớn
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obj_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:  # Chỉ khoanh vùng vật thể có diện tích đủ lớn (>1000 pixel)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                obj_count += 1

        # Lưu cả ảnh kết quả khoanh vùng và ảnh trắng đen (threshold) để đối chiếu
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"contour_{filename}"), img_resized)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"thresh_{filename}"), thresh)
        print(f"- {filename}: Phát hiện và khoanh vùng {obj_count} cụm đối tượng chính.")

    print(f"=> Hoàn tất! Kiểm tra kết quả trong thư mục '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    process_classical_cv()