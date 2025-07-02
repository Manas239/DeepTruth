import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Load the trained model
model = tf.keras.models.load_model("models/deepfake_image_model.h5")

# Predict function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Fake" if prediction >= 0.5 else "Real"
    confidence = round(prediction * 100, 2) if label == "Fake" else round((1 - prediction) * 100, 2)
    
    print(f"\nPrediction: {label} ({confidence}% confidence)")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_image(sys.argv[1])

#python predict.py test.jpg
