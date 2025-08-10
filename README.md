# BERT Sentiment Analysis App

A web application that uses BERT (Bidirectional Encoder Representations from Transformers) to analyze the sentiment of text input.

## Features

- **Real-time sentiment analysis** using Auto model
- **28 emotions** Classification
- **Confidence scores** for each sentiment level
- **User-friendly interface** with emoji indicators
- **Multilingual support** 

## How to Use

1. Enter your text in the text area
2. Click "Analyze Sentiment" button
3. View the predicted sentiment rating and confidence scores



## Technical Details

- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Framework**: Flask for backend, React for frontend
- **ML Library**: Transformers (Hugging Face)
- **Deep Learning**: PyTorch

## Local Development

To run this app locally:

```bash
#For python
pip install -r requirements.txt
python app.py
#For React
npm init -y
npm start
```
