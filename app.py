# Install dependencies before running:
# pip install streamlit transformers torch

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# --------------------
# 1️⃣ Load Model & Tokenizer
# --------------------
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
# For binary sentiment use: distilbert-base-uncased-finetuned-sst-2-english

@st.cache_resource  # Cache model so it doesn't reload every time
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# --------------------
# 2️⃣ Streamlit UI
# --------------------
st.title("📊 BERT Sentiment Analysis")
st.write("Type a sentence below and see the sentiment prediction (1–5 stars).")

# Text input
user_input = st.text_area("📝 Enter text here", "I love this product! It's amazing.")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # --------------------
        # 3️⃣ Run inference
        # --------------------
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        sentiment_score = predicted_class_id + 1  # 1–5 stars

        # --------------------
        # 4️⃣ Display Result with Labels
        # --------------------
        sentiment_labels = {
            1: "😠 Very Negative",
            2: "🙁 Negative",
            3: "😐 Neutral",
            4: "🙂 Positive",
            5: "🤩 Very Positive"
        }

        sentiment_label = sentiment_labels.get(sentiment_score, "Unknown")

        st.success(f"🌟 Predicted Sentiment: **{sentiment_score} Stars - {sentiment_label}**")

        # Optional: Show probabilities
        probs = torch.softmax(logits, dim=1).flatten().tolist()
        st.write("Confidence Scores:")
        for i, p in enumerate(probs):
            st.write(f"{i+1} ⭐ ({sentiment_labels[i+1]}): {p*100:.2f}%")
