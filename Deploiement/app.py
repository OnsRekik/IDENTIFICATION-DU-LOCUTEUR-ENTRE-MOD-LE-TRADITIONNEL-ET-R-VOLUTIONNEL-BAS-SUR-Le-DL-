from flask import Flask, request, render_template
import numpy as np
import librosa
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import tensorflow as tf
from tensorflow.keras.models import load_model  # Import load_model from tensorflow.keras
import joblib




# Hyperparameters
n_mfcc = 13
max_pad_len = 100

# Load your trained model
model = load_model("speaker_cnn_model.keras")
label_encoder = joblib.load("label_encoder.pkl")


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_mfcc_features(audio, sample_rate, n_mfcc=n_mfcc, max_pad_len=max_pad_len):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    if mfccs is None or len(mfccs) == 0:
        return None
    mfccs = np.pad(mfccs, ((0, 0), (0, max(0, max_pad_len - mfccs.shape[1]))), mode='constant')
    return mfccs[:, :max_pad_len]

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = extract_mfcc_features(audio, sr)
    if mfcc is None:
        return None
    mfcc = mfcc[..., np.newaxis]              # shape: (13, 100, 1)
    return np.expand_dims(mfcc, axis=0)       # shape: (1, 13, 100, 1)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        input_data = preprocess_audio(filepath)
        if input_data is not None:
            preds = model.predict(input_data)
            speaker_id = int(np.argmax(preds))
            speaker_label = label_encoder.inverse_transform([speaker_id])[0]
            prediction = f"Predicted Speaker: {speaker_label}"
        else:
            prediction = "Error: Could not process the audio file."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
