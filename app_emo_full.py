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
        emo_long=["Admiration","Amusement","Anger","Annoyance","Approval","Caring","Confusion","Curiosity",
                  "Desire","Disappointment","Disapproval","Disgust","Embarrassment","Excitement","Fear",
                  "Gratitude","Grief","Joy","Love","Nervousness","Optimism","Pride","Realization","Relief",
                  "Remorse","Sadness","Surprise","Neutral"]
        emo_dict_long = dict(zip(emo_long, probs))
        emo = pd.DataFrame(emo_dict_long, index=['Emotions']).T

        emo = emo.reset_index()
        emo.columns = ["Emotion", "Percentage"]

        emo_sorted = emo.sort_values(by='Percentage', ascending=False)
        emo_sorted = emo[emo['Percentage'] > 1]


        # emo.fillna(0, inplace=True)
        # emo_dict = emo.sum(axis=0).to_dict()
        # emo_dict = {k: round(v, 2) for k, v in emo_dict.items()}



        sentiment_score=emo.loc[emo['Percentage'].idxmax()]['Emotion']
        sentiment_label = emo.loc[emo['Percentage'].idxmax()]['Percentage']




        st.success(f"🌟 Predicted Sentiment: **{sentiment_score}%- {sentiment_label}**")

        # Optional: Show probabilities
        # probs = torch.softmax(logits, dim=1).flatten().tolist()
        st.write("Confidence Scores:")
        for index, row in emo_sorted.iterrows():
            st.write(f" {row['Emotion']} : {row['Percentage']:.2f}%")
