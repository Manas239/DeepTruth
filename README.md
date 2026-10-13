# DeepTruth

DeepTruth is a deepfake image detection project that uses transfer learning with MobileNetV2 to classify images as Real or Fake. The repository includes a training pipeline, a CLI prediction tool, and a Streamlit web app for interactive image analysis.

## Overview

This project was built to help detect AI-generated or manipulated face images by training a lightweight convolutional neural network on a binary real-vs-fake classification task. It is designed for quick experimentation, model training, and prediction in both command-line and browser-based workflows.

## Features

- MobileNetV2-based transfer learning model
- Binary classification for Real vs Fake images
- Streamlit web interface for image upload and prediction
- CLI prediction script for single-image inference
- Prediction history stored in JSON for quick review
- Dataset organization utility for preparing image folders

## Tech Stack

- Python 3
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- Pillow
- NumPy
- Pandas
- Streamlit
- scikit-learn

## Project Structure

```text
DeepTruth/
├── app.py                  # Streamlit web application
├── train.py                # Model training pipeline
├── predict.py              # CLI prediction script
├── organize_images.py      # Dataset organization helper
├── requirements.txt        # Python dependencies
├── history.json            # Stores prediction history
├── data/                   # Training dataset folder
│   ├── real/               # Real image samples
│   └── fake/               # Fake/deepfake image samples
├── models/                 # Trained model output directory
│   └── deepfake_image_model.h5
├── archive/                # Archived or raw dataset files
├── test.jpg                # Sample image for quick testing
├── test1.jpg               # Additional sample image
├── test2.jpg               # Additional sample image
├── test3.jpg               # Additional sample image
├── test4.jpg               # Additional sample image
├── test5.jpg               # Additional sample image
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules (if present)
```

## Dataset Format

The training script expects the dataset folder structure below:

```text
data/
├── real/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── fake/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```

You can organize your dataset manually or use the included `organize_images.py` script if needed.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Manas239/DeepTruth.git
cd DeepTruth
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

Run:

```bash
python train.py
```

This will:

- load images from `data/`
- split the dataset into train/validation sets
- fine-tune a MobileNetV2 model
- save the trained model to `models/deepfake_image_model.h5`

## Running the Web App

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

The app lets you upload an image and view the prediction result along with a lightweight history of prior predictions.

## Running Predictions from the Command Line

```bash
python predict.py test.jpg
```

Example output:

```text
Prediction: Real (98.45% confidence)
```

## How Prediction Works

The model takes an image, resizes it to 224x224, normalizes the pixel values, and passes it through MobileNetV2. The final layer outputs a sigmoid probability. If the value is above 0.5, the image is classified as Fake; otherwise it is labeled Real.

## Notes

- The app reads and writes prediction history to `history.json`.
- The trained model is expected under `models/deepfake_image_model.h5` before running inference.
- GPU support is optional; the project can run on CPU as well.

## License

This project is provided as-is for educational and research use. Please check the repository for the applicable license details before using it in production or redistributing it.

## Author

Manas239

## Contact

For questions, suggestions, or collaboration opportunities, reach out via the repository or project profile on GitHub.
