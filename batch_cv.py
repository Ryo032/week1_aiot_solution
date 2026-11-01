import cv2
import os
import glob

# Đường dẫn thư mục
INPUT_DIR = "images_raw"
OUTPUT_DIR = "images_processed"

def process_images():
    print("=== BẮT ĐẦU XỬ LÝ LÔ ẢNH (OPENCV) ===")
    
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Tìm tất cả file .jpg trong thư mục raw
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    
    if not image_paths:
        print(f"[LỖI] Không tìm thấy ảnh .jpg nào trong thư mục '{INPUT_DIR}'!")
        return

    # Lặp qua từng ảnh để xử lý
    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        print(f"Đang xử lý: {file_name}...")
        
        # 1. Đọc ảnh bằng OpenCV
        img = cv2.imread(img_path)

        
        
        if img is None:
            print(f"  -> Bỏ qua file lỗi: {file_name}")
            continue

        # 2. Thay đổi kích thước (Resize) về chuẩn 640x640 pixel
        img_resized = cv2.resize(img, (640, 640))
        
        # 3. Chuyển sang ảnh xám (Grayscale) để giảm tải tính toán
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 4. Ghi ảnh đã xử lý ra thư mục output
        out_path = os.path.join(OUTPUT_DIR, file_name)
        cv2.imwrite(out_path, img_gray)
        
    print(f"\n=== ĐÃ XỬ LÝ XONG {len(image_paths)} ẢNH ===")
    print(f"Hãy kiểm tra thư mục: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_images()