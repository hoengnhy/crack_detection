import tensorflow as tf

# 🔹 Kiểm tra và kích hoạt GPU nếu có
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Bật "memory growth" để tránh lỗi thiếu RAM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU ĐÃ KÍCH HOẠT: {gpus[0]}")
    except RuntimeError as e:
        print(f"❌ LỖI GPU: {e}")
else:
    print("⚠️ KHÔNG TÌM THẤY GPU. SẼ DÙNG CPU!")

# 🔹 Cấu hình dataset
IMAGE_DIR = "dataset/train_img"
MASK_DIR = "dataset/train_lab"

# 🔹 Kích thước ảnh input
IMG_SIZE = 256

# 🔹 Số epoch huấn luyện
EPOCHS = 100

# 🔹 Batch size
BATCH_SIZE = 8

# 🔹 Đường dẫn lưu mô hình
MODEL_PATH = "models/unet_crack_detection.h5"

# 🔹 Kiểm tra lại danh sách thiết bị TensorFlow nhận diện
print("💻 THIẾT BỊ ĐƯỢC TENSORFLOW NHẬN DIỆN:")
print(tf.config.list_physical_devices())
