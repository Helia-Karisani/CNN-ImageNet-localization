# CNN-ImageNet-localization

## 🧠 Object Localization with PyTorch

This project demonstrates how to build and train a **Convolutional Neural Network (CNN)** for **object localization** — predicting the bounding box coordinates of an object in an image.
It uses a pre-trained **EfficientNet-b0** backbone from the `timm` library and fine-tunes it for bounding box regression.

---

### 📂 Repository Structure

```
.
├── CNN_PyTorch_object_localization.ipynb   # Main Colab notebook
├── train.csv                               # Training labels (image filenames + bounding boxes)
├── utils.py                                # Helper functions for data loading and visualization
├── best_model.pt                           # Saved trained model weights
├── train_images.zip                        # Zipped training images folder
└── README.md                               # (This file)
```

---

### 🚀 How to Run

You can run this project directly in **Google Colab** or locally with Python.

#### **1️⃣ Setup**

Install required packages:

```bash
!pip install timm
!pip install albumentations
!pip install opencv-python-headless
```

#### **2️⃣ Unzip dataset**

If you’re using the GitHub version:

```bash
!unzip -q train_images.zip -d object-localization-dataset/
```

#### **3️⃣ Open notebook**

Run all cells in:

```
CNN_PyTorch_object_localization.ipynb
```

The notebook handles:

* Data loading and augmentation
* Model definition (`EfficientNet_b0`)
* Training and loss computation
* Evaluation and visualization of bounding boxes

---

### 📊 Dataset

* The dataset is stored in `train_images.zip`
* Each image’s bounding box coordinates are stored in `train.csv`
  (columns typically include `x_min`, `y_min`, `x_max`, `y_max`).

---

### 🧩 Model

* **Backbone:** EfficientNet_b0 (from `timm`)
* **Task:** Regression (predict bounding box coordinates)
* **Loss Function:** Mean Squared Error (MSE)

---

### 🏁 Output

* The trained model is saved as `best_model.pt`.
* The notebook visualizes predicted bounding boxes on test images.

---

### 📘 References

* [EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
* [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)
* [Albumentations Documentation](https://albumentations.ai/docs/)
