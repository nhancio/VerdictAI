# Install dependencies before running:
# pip install streamlit transformers torch

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------
# 1️⃣ Load Model & Tokenizer
# --------------------
MODEL_NAME = "sangkm/go-emotions-fine-tuned-distilroberta"
# For binary sentiment use: distilbert-base-uncased-finetuned-sst-2-english

@st.cache_resource  # Cache model so it doesn't reload every time
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
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

        outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1).flatten().tolist()
        probs = [round(i * 100, 2) for i in probs]
        #
        # logits = outputs.logits
        # predicted_class_id = torch.argmax(logits, dim=1).item()
        # sentiment_score = predicted_class_id + 1  # 1–5 stars

        #mapping the sentiments to relevant ones
        emo_long = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
                    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
                    "remorse", "sadness", "surprise", "neutral"]
        emo_dict_long = dict(zip(emo_long, probs))
        emotion_map = {
            "joy": "Pleasant",
            "love": "Pleasant",
            "gratitude": "Pleasant",
            "amusement": "Pleasant",
            "approval": "Pleasant",
            "optimism": "Pleasant",
            "admiration": "Pleasant",
            "caring": "Pleasant",
            "pride": "Pleasant",
            "relief": "Pleasant",

            "anger": "Aggression",
            "disgust": "Aggression",
            "annoyance": "Aggression",
            "disapproval": "Aggression",

            "fear": "Anxious",
            "nervousness": "Anxious",

            "remorse": "Sadness",
            "disappointment": "Sadness",
            "grief": "Sadness",
            "sadness": "Sadness",
            "embarrassment": "Sadness",

            "surprise": "Neutral",
            "realization": "Neutral",
            "confusion": "Neutral",
            "curiosity": "Neutral",
            "neutral": "Neutral",

            "desire": "Neutral",
            "excitement": "Neutral"
        }
        emo_dict_mapped = [{emotion_map[key]: value} for key, value in emo_dict_long.items()]
        emo = pd.DataFrame(emo_dict_mapped)
        emo.fillna(0, inplace=True)
        emo_dict = emo.sum(axis=0).to_dict()
        emo_dict = {k: round(v, 2) for k, v in emo_dict.items()}
        sentiment_score=max(emo_dict.values())
        sentiment_label = max(emo_dict, key=emo_dict.get)
        # sentiment_label = emo_dict.get(keys[0], "Unknown")

        # # --------------------
        # # 4️⃣ Display Result with Labels
        # # --------------------
        # sentiment_labels = {
        #     1: "😠 Aggression",
        #     2: "🙁 Sadness",
        #     3: "😐 Anxious",
        #     4: "🙂 Neutral",
        #     5: "🤩 Pleasant"
        # }

        # sentiment_label = sentiment_labels.get(sentiment_score, "Unknown")

        st.success(f"🌟 Predicted Sentiment: **{sentiment_score}%- {sentiment_label}**")

        # Optional: Show probabilities
        # probs = torch.softmax(logits, dim=1).flatten().tolist()
        st.write("Confidence Scores:")
        for key,value in emo_dict.items():
            st.write(f" {key} : {value:.2f}%")
