from dataset_preprocessing import load_dataset
from unet_model import build_unet
from config import MODEL_PATH, EPOCHS, BATCH_SIZE

# Load dữ liệu
X_train, X_test, Y_train, Y_test = load_dataset()

# Khởi tạo mô hình
model = build_unet()

print("🔍 Kiến trúc mô hình U-Net:")
model.summary()

# Huấn luyện
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Lưu mô hình
model.save(MODEL_PATH)
print("✅ Mô hình đã được lưu!")
