# BERT Sentiment Analysis App

A Streamlit web application that uses BERT (Bidirectional Encoder Representations from Transformers) to analyze the sentiment of text input.

## Features

- **Real-time sentiment analysis** using BERT model
- **5-star rating system** (1-5 stars)
- **Confidence scores** for each sentiment level
- **User-friendly interface** with emoji indicators
- **Multilingual support** (using multilingual BERT model)

## How to Use

1. Enter your text in the text area
2. Click "Analyze Sentiment" button
3. View the predicted sentiment rating and confidence scores

## Sentiment Ratings

- ⭐⭐⭐⭐⭐ (5 stars): 🤩 Very Positive
- ⭐⭐⭐⭐ (4 stars): 🙂 Positive  
- ⭐⭐⭐ (3 stars): 😐 Neutral
- ⭐⭐ (2 stars): 🙁 Negative
- ⭐ (1 star): 😠 Very Negative

## Technical Details

- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Framework**: Streamlit
- **ML Library**: Transformers (Hugging Face)
- **Deep Learning**: PyTorch

## Local Development

To run this app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

This app is deployed on Streamlit Cloud and can be accessed at: [Your Streamlit Cloud URL will appear here after deployment] 