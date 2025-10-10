from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    from keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Paths for models & tokenizers
model_a_path = r"C:\Users\SIDHARTH\OneDrive\python\flask1\models\Model_A.h5"
model_b_path = r"C:\Users\SIDHARTH\OneDrive\python\flask1\models\Model_B1.h5"
tk_path = r"C:\Users\SIDHARTH\OneDrive\python\flask1\models\tokenizer.pkl"
tk2_path = r"C:\Users\SIDHARTH\OneDrive\python\flask1\models\tokenizer1.pkl"
le_path = r"C:\Users\SIDHARTH\OneDrive\python\flask1\models\label_encoder.pkl"
le2_path = r"C:\Users\SIDHARTH\OneDrive\python\flask1\models\label_encoder1.pkl"

# Load models once
if not os.path.exists(model_a_path) or not os.path.exists(model_b_path):
    raise FileNotFoundError("❌ Model files not found. Check paths again.")

model_a = tf.keras.models.load_model(model_a_path)
model_b = tf.keras.models.load_model(model_b_path)

# Load tokenizers safely
with open(tk_path, "rb") as f:
    tokenizer_a = pickle.load(f)
with open(tk2_path, "rb") as f:
    tokenizer_b = pickle.load(f)

# Load label encoders safely
with open(le_path, "rb") as f:
    label_enc_a = pickle.load(f)
with open(le2_path, "rb") as f:
    label_enc_b = pickle.load(f)

# Max sequence lengths (must match training settings)
max_len_a = 100
max_len_b = 120

# Initialize Flask app
app = Flask(__name__)

# Helper function: Display stars
def display_stars(rating, max_stars=5):
    full_star = "⭐"
    empty_star = "☆"
    return full_star * int(rating) + empty_star * (max_stars - int(rating))

# Home page route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction_a, prediction_b = None, None
    probs_a, probs_b = None, None
    stars_a, stars_b = "", ""
    review_text = ""
    error = None

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()

        # Validation: no empty or numeric-only input
        if not review_text or review_text.isdigit() or not any(c.isalpha() for c in review_text):
            error = "❌ Please enter a valid review (no numbers/gibberish)."
        else:
            try:
                # ---------- Model A ----------
                seq_a = tokenizer_a.texts_to_sequences([review_text])
                pad_a = pad_sequences(seq_a, maxlen=max_len_a, padding="post")
                pred_a = model_a.predict(pad_a, verbose=0)[0]  # array of probs
                prediction_a = label_enc_a.inverse_transform([np.argmax(pred_a)])[0]
                stars_a = display_stars(prediction_a)
                probs_a = {label_enc_a.classes_[i]: float(f"{p*100:.2f}") for i, p in enumerate(pred_a)}

                # ---------- Model B ----------
                seq_b = tokenizer_b.texts_to_sequences([review_text])
                pad_b = pad_sequences(seq_b, maxlen=max_len_b, padding="post")
                pred_b = model_b.predict(pad_b, verbose=0)[0]
                prediction_b = label_enc_b.inverse_transform([np.argmax(pred_b)])[0]
                stars_b = display_stars(prediction_b)
                probs_b = {label_enc_b.classes_[i]: float(f"{p*100:.2f}") for i, p in enumerate(pred_b)}

            except Exception as e:
                error = f"⚠ Error during prediction: {str(e)}"

    return render_template(
        "index.html",
        review=review_text,
        pred_a=prediction_a,
        pred_b=prediction_b,
        stars_a=stars_a,
        stars_b=stars_b,
        probs_a=probs_a,
        probs_b=probs_b,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)