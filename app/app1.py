import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder

# ================================
# Paths for models & tokenizers
# ================================
base_path = r"C:\Users\SIDHARTH\OneDrive\python\flask\models"

model_a_path = os.path.join(base_path, "Model_A.h5")
model_b_path = os.path.join(base_path, "Model_B.h5")
tk_path = os.path.join(base_path, "tokenizer.pkl")
tk2_path = os.path.join(base_path, "tokenizer2.pkl")
le_path = os.path.join(base_path, "label_encoder.pkl")
le2_path = os.path.join(base_path, "label_encoder2.pkl")

# ================================
# Check if files exist
# ================================
for f in [model_a_path, model_b_path, tk_path, tk2_path, le_path, le2_path]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"❌ Missing required file: {f}")

# ================================
# Load models
# ================================
model_a = tf.keras.models.load_model(model_a_path)
model_b = tf.keras.models.load_model(model_b_path)

# ================================
# Load tokenizers
# ================================
with open(tk_path, "rb") as f:
    tokenizer_a = pickle.load(f)
with open(tk2_path, "rb") as f:
    tokenizer_b = pickle.load(f)

# ================================
# Load label encoders
# ================================
with open(le_path, "rb") as f:
    label_enc_a = pickle.load(f)
with open(le2_path, "rb") as f:
    label_enc_b = pickle.load(f)

# ================================
# Flask app setup
# ================================
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["review"]

        # Preprocess input for Model A
        seq_a = tokenizer_a.texts_to_sequences([text])
        padded_a = pad_sequences(seq_a, maxlen=100, padding="post")
        pred_a = model_a.predict(padded_a)
        result_a = label_enc_a.inverse_transform([np.argmax(pred_a)])

        # Preprocess input for Model B
        seq_b = tokenizer_b.texts_to_sequences([text])
        padded_b = pad_sequences(seq_b, maxlen=100, padding="post")
        pred_b = model_b.predict(padded_b)
        result_b = label_enc_b.inverse_transform([np.argmax(pred_b)])

        return render_template(
            "index.html",
            input_text=text,
            prediction_a=result_a[0],
            prediction_b=result_b[0]
        )

# ================================
# Run app
# ================================
if __name__ == "__main__":
    app.run(debug=True)
