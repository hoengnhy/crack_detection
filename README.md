# 🚀 Crack Detection - Detect Cracks with Deep Learning

## 📌 File Execution Order

| **Step** | **Command** | **Description** |
|----------|------------|-----------------|
| 1️⃣ | `pip install -r requirements.txt` | Install required libraries |
| 2️⃣ | `python dataset_preprocessing.py` | Preprocess the dataset (if needed) |
| 3️⃣ | `python train.py` | Train the U-Net model |
| 4️⃣ | `python predict.py --image_path path/to/image.jpg` | Predict cracks in an image |

📌 **Note:** Replace `path/to/image.jpg` with the actual image path.

---

## 📂 Project Structure

```
├── dataset/                # Training & testing dataset
├── models/                 # Trained model files (.h5)
├── config.py               # Model configuration settings
├── dataset_preprocessing.py # Dataset preprocessing script
├── train.py                # Model training script
├── predict.py              # Crack detection prediction script
├── unet_model.py           # U-Net model definition
├── requirements.txt        # Required dependencies
└── README.md               # Project documentation
```
