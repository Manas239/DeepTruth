# 🧠 Deepfake Image Detector

A deep learning-based project to detect whether an image is **real or deepfake** using a Convolutional Neural Network (CNN).  
This project uses **transfer learning with MobileNetV2**, and was built from scratch using a labeled dataset of real and fake faces.

---

## 📌 Project Overview

Deepfakes are AI-generated fake content that can spread misinformation.  
This project aims to build a simple yet effective system to identify whether an image is **real** or **fake** (deepfake) using computer vision and machine learning.

---

## ✅ Features / Functionalities

- 📂 Image Dataset Organization (Real & Fake classes)
- 🧠 CNN-based Image Classifier using **MobileNetV2**
- 📈 Model Training & Evaluation
- 🖼️ Real-time Prediction with Image Display
- 💾 Saves Trained Model (`.h5` format)
- 🧪 CLI tool to test any image from your local system

---

## 🛠️ Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DeepfakeImageDetector.git
cd DeepfakeImageDetector
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

> Or install manually:

```bash
pip install tensorflow numpy matplotlib
```

### 3. Organize the Dataset

Download the dataset from Kaggle:  
[Real vs Fake Face Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

Structure it like this:

```
DeepfakeImageDetector/
├── data/
│   ├── real/      # real_0001.jpg, real_0002.jpg ...
│   └── fake/      # fake_0001.jpg, fake_0002.jpg ...
```

You can also run the provided `organize_images.py` script to move images automatically.

### 4. Train the Model

```bash
python train.py
```

After training, the model is saved as:

```
models/deepfake_image_model.h5
```

### 5. Predict a Test Image

Place your test image (e.g., `test.jpg`) in the project folder and run:

```bash
python predict.py test.jpg
```

A pop-up will show the image with prediction (Real or Fake).

---

## 📦 Project Structure

```
DeepfakeImageDetector/
├── data/
│   ├── real/
│   └── fake/
├── models/
│   └── deepfake_image_model.h5
├── train.py
├── predict.py
├── organize_images.py
├── requirements.txt
└── README.md
```

---

## 🧠 Model Used

- **MobileNetV2** (pretrained on ImageNet)
- Used as a feature extractor (transfer learning)
- Custom classification head:
  - GlobalAveragePooling2D
  - Dense(64, relu)
  - Dense(1, sigmoid)

---

## 🚀 Future Scope

- 🔄 Extend to detect **deepfake videos**
- 🎙️ Add support for **deepfake audio detection**
- 🌐 Build a **web app** using Flask or Streamlit
- 📱 Deploy as a mobile app for real-time use
- 📊 Add model explainability with Grad-CAM or LIME

---

## 🌟 Advantages

- ✅ Lightweight and fast model
- ✅ Works well on limited datasets
- ✅ Simple to use and extend
- ✅ Detects deepfakes at high accuracy
- ✅ Educational and beginner-friendly

---

## 🧑‍💻 Developed By

**Deepika Sharma**  
Aspiring Machine Learning(ML) | Deep Learning Enthusiast

**Manas Doltade**
Aspiring Data Science(DS) | Machine Learning Enthunsiast

---
## 📬 Feedback / Contributions

Feel free to raise an issue or submit a pull request if you have ideas or improvements!
