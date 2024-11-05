# Advanced Trading Bot

A machine learning-powered trading analysis dashboard built with Python and Streamlit. Combines technical analysis, sentiment analysis, and LSTM price predictions to generate trading signals.

## Key Features
- Real-time technical indicator calculations (RSI, MACD, Bollinger Bands)
- News sentiment analysis using TextBlob
- Price predictions using LSTM neural networks
- Interactive dashboard with candlestick charts and indicator visualizations
- SQLite database for tracking stock watchlist and analysis history

## Tech Stack
- Python 3.10+
- Streamlit
- PyTorch (LSTM)
- yfinance
- pandas-ta
- plotly

## Installation
```bash
pip install -r requirements.txt
streamlit run advisor.py
