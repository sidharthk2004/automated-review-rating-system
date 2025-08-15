import streamlit as st
import joblib


def display_stars(rating, max_stars=5):
    full_star = "⭐"
    empty_star = "☆"
    return full_star * int(rating) + empty_star * (max_stars - int(rating))

model_a = joblib.load("model A.pkl")
vectorizer_a = joblib.load("vectorizer.pkl")

model_b = joblib.load("model B.pkl")
vectorizer_b = joblib.load("vectorizer2.pkl")


st.set_page_config(page_title="Review Rating Comparator", layout="centered")
st.title("Review Score Estimator: Model A vs Model B")

st.markdown("""
Prediction Battle  🤖⚔️🤖:
-  Model A: 📉 *Trained on Class-Imbalanced Data*
-  Model B: 📈 *Trained on Class-Balanced data*
""")


review_text = st.text_area("🖋️ Review dropbox 📦:")

if st.button("Predict"):
    if not review_text.strip():
        st.warning("Please enter a review before predicting.")
    else:
        
        X_a = vectorizer_a.transform([review_text])
        X_b = vectorizer_b.transform([review_text])

        
        pred_a = model_a.predict(X_a)[0]
        pred_b = model_b.predict(X_b)[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model A (Imbalanced)")
            st.write(f"Predicted Rating: {pred_a}")
            st.markdown(display_stars(pred_a))

        with col2:
            st.subheader("Model B (Balanced)")
            st.write(f"Predicted Rating: {pred_b}")
            st.markdown(display_stars(pred_b))
