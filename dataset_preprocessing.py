import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from config import IMAGE_DIR, MASK_DIR, IMG_SIZE

def load_dataset():
    X, Y = [], []
    
    image_files = sorted(os.listdir(IMAGE_DIR))
    mask_files = sorted(os.listdir(MASK_DIR))

    for img_name, mask_name in zip(image_files, mask_files):
        img = cv2.imread(os.path.join(IMAGE_DIR, img_name))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Chuẩn hóa pixel
        X.append(img)

        mask = cv2.imread(os.path.join(MASK_DIR, mask_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = mask / 255.0  # Chuẩn hóa
        mask = np.expand_dims(mask, axis=-1)  # Thêm chiều cho mask
        Y.append(mask)

    X = np.array(X)
    Y = np.array(Y)

    return train_test_split(X, Y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_dataset()
    print(f"Số lượng ảnh train: {len(X_train)}, test: {len(X_test)}")
