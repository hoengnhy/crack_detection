import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from threading import Thread
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import os
from config import MODEL_PATH, IMG_SIZE

# Load mô hình
model = load_model(MODEL_PATH)

# Biến dừng tất cả loại detect
stop_detection = False

# Hàm hiển thị ảnh gốc và ảnh detect
def process_and_display(image, prediction, window_name="Result"):
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = (prediction.squeeze() * 255).astype(np.uint8)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Ghép ảnh gốc và mask cạnh nhau
    combined = cv2.hconcat([image_resized, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Hàm dự đoán trên ảnh
def predict_image(image_path):
    global stop_detection
    if stop_detection:
        return
    
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    prediction = (prediction > 0.5).astype(np.uint8)

    original = cv2.imread(image_path)
    process_and_display(original, prediction, window_name="Image Detection")

# Hàm nhận diện vết nứt từ video (hiển thị ảnh gốc và mask cùng lúc)
def predict_video(video_path):
    global stop_detection
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        if stop_detection:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(frame_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        prediction = (prediction > 0.5).astype(np.uint8)

        mask = (prediction.squeeze() * 255).astype(np.uint8)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        # Ghép ảnh gốc và mask cạnh nhau
        combined = cv2.hconcat([frame_resized, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

        cv2.imshow("Video Crack Detection", combined)

        if cv2.waitKey(10) & 0xFF == ord('q') or stop_detection:
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm chạy camera detection
def predict_camera():
    global stop_detection
    stop_detection = False

    cap = cv2.VideoCapture(0)  # Mở camera

    while not stop_detection:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(frame_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        prediction = (prediction > 0.5).astype(np.uint8)

        mask = (prediction.squeeze() * 255).astype(np.uint8)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        # Ghép ảnh gốc và mask cạnh nhau
        combined = cv2.hconcat([frame_resized, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

        cv2.imshow("Real-Time Crack Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord('q') or stop_detection:
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm thực hiện inference trên toàn bộ thư mục ảnh
def predict_folder():
    global stop_detection
    stop_detection = False

    folder_path = filedialog.askdirectory()
    if not folder_path:
        return  # Người dùng bấm cancel

    image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))]
    
    if not image_files:
        print("❌ Không có ảnh nào trong thư mục này!")
        return

    for image_name in image_files:
        if stop_detection:
            break
        image_path = os.path.join(folder_path, image_name)
        print(f"🔍 Đang xử lý: {image_path}")
        predict_image(image_path)

# Hàm dừng tất cả các loại detection
def stop_all_detection():
    global stop_detection
    stop_detection = True
    print("🛑 Đã dừng tất cả quá trình nhận diện!")

# Hàm chọn file từ máy (Ảnh & Video chung)
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image & Video files", "*.jpg;*.png;*.mp4;*.avi")])
    if file_path.endswith((".jpg", ".png")):
        Thread(target=predict_image, args=(file_path,)).start()
    elif file_path.endswith((".mp4", ".avi")):
        Thread(target=predict_video, args=(file_path,)).start()

# Giao diện chọn chế độ
def main():
    root = tk.Tk()
    root.title("Crack Detection")
    root.geometry("400x300")

    tk.Label(root, text="Chọn chế độ nhận diện vết nứt", font=("Arial", 12)).pack(pady=10)

    btn_camera = tk.Button(root, text="📷 Camera Detection", command=lambda: Thread(target=predict_camera).start(), font=("Arial", 10))
    btn_camera.pack(pady=5)

    btn_file = tk.Button(root, text="📂 Load Image/Video", command=open_file, font=("Arial", 10))
    btn_file.pack(pady=5)

    btn_folder = tk.Button(root, text="📂 Inference Folder", command=lambda: Thread(target=predict_folder).start(), font=("Arial", 10))
    btn_folder.pack(pady=5)

    btn_stop = tk.Button(root, text="🛑 STOP DETECTION", command=stop_all_detection, font=("Arial", 10), fg="red")
    btn_stop.pack(pady=5)

    btn_exit = tk.Button(root, text="❌ Thoát", command=root.quit, font=("Arial", 10))
    btn_exit.pack(pady=5)

    root.mainloop()

# Chạy chương trình
if __name__ == "__main__":
    main()
