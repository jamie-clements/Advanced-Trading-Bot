# 🤖 Advanced Trading Bot

A sophisticated trading analysis dashboard that combines machine learning, technical analysis, and sentiment analysis to generate high-confidence trading signals. Built with Python and Streamlit, this platform provides a comprehensive suite of tools for market analysis and trading decisions.

## ✨ Key Features

- **Technical Analysis Engine**
  - 15+ technical indicators including RSI, MACD, Bollinger Bands
  - Custom pattern recognition for candlestick formations
  - Automated signal generation with confidence scoring

- **Machine Learning Integration**
  - LSTM neural networks for price predictions
  - Custom-trained models for each stock
  - Rolling prediction accuracy tracking

- **Sentiment Analysis**
  - Real-time news sentiment tracking
  - TextBlob-powered natural language processing
  - Sentiment trend analysis and impact scoring

- **Interactive Dashboard**
  - Real-time candlestick charts with technical overlays
  - Comprehensive stock watchlist management
  - Historical analysis tracking and performance metrics
  - Custom visualization for all indicators and signals

## 🛠️ Tech Stack

- **Core**: Python 3.10+
- **Framework**: Streamlit
- **Data Processing**: 
  - pandas
  - numpy
  - yfinance
  - pandas-ta
- **Machine Learning**: 
  - PyTorch
  - scikit-learn
- **Visualization**: 
  - Plotly
  - Streamlit components
- **Storage**: SQLite3

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/advanced-trading-bot.git
cd advanced-trading-bot
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run advisor.py
```

## 📊 Usage

1. Add stocks to your watchlist using the sidebar
2. View real-time analysis across four main sections:
   - Active Signals: Current trading recommendations
   - Technical Analysis: Detailed indicator charts
   - News Impact: Sentiment analysis and news tracking
   - Performance: Historical accuracy and predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License 

## ⚠️ Disclaimer

This software is for educational and research purposes only. Do not use this system for actual trading without understanding the risks involved. Always do your own research and consult with financial advisors before making investment decisions.
