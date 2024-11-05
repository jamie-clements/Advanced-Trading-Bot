from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
import requests
import sqlite3
import ta
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import logging
import json
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    symbol: str
    date: datetime
    type: str  # 'buy' or 'sell'
    price: float
    quantity: int
    total: float
    confidence: float
    signals: Dict[str, str]
    notes: str = ""

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

class TechnicalAnalysis:
    def __init__(self):
        self.pattern_recognition = self._setup_pattern_recognition()
        
    def _setup_pattern_recognition(self):
        """Initialize candlestick pattern recognition"""
        patterns = {
            'doji': self._is_doji,
            'hammer': self._is_hammer,
            'engulfing': self._is_engulfing,
            'morning_star': self._is_morning_star
        }
        return patterns
    
    @staticmethod
    def _is_doji(candle: pd.Series, threshold: float = 0.1) -> bool:
        """Check for Doji pattern"""
        body = abs(candle['Open'] - candle['Close'])
        upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])
        lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
        return body <= threshold * (upper_shadow + lower_shadow)

    @staticmethod
    def _is_hammer(candle: pd.Series) -> bool:
        """Check for Hammer pattern"""
        body = abs(candle['Open'] - candle['Close'])
        upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])
        lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
        return lower_shadow > 2 * body and upper_shadow < body

    @staticmethod
    def _is_engulfing(candles: pd.DataFrame, idx: int) -> bool:
        """Check for Engulfing pattern"""
        if idx == 0:
            return False
        curr = candles.iloc[idx]
        prev = candles.iloc[idx-1]
        return (curr['Open'] > prev['Close'] and curr['Close'] < prev['Open']) or \
               (curr['Open'] < prev['Close'] and curr['Close'] > prev['Open'])

    @staticmethod
    def _is_morning_star(candles: pd.DataFrame, idx: int) -> bool:
        """Check for Morning Star pattern"""
        if idx < 2:
            return False
        first = candles.iloc[idx-2]
        second = candles.iloc[idx-1]
        third = candles.iloc[idx]
        return first['Close'] < first['Open'] and \
               abs(second['Open'] - second['Close']) < abs(first['Open'] - first['Close']) * 0.1 and \
               third['Close'] > third['Open']

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        # Trend Indicators
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        
        # Momentum Indicators
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['Close'])
        df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        
        # Volatility Indicators
        df['bbands_upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['bbands_lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Volume Indicators
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Pattern Recognition
        df['doji'] = df.apply(self._is_doji, axis=1)
        df['hammer'] = df.apply(self._is_hammer, axis=1)
        df['engulfing'] = pd.Series([self._is_engulfing(df, i) for i in range(len(df))])
        df['morning_star'] = pd.Series([self._is_morning_star(df, i) for i in range(len(df))])
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """Generate trading signals with confidence levels"""
        signals = {}
        confidence_scores = []
        
        # RSI Signals (0-100 scale)
        rsi = df['rsi'].iloc[-1]
        if rsi < 30:
            signals['RSI'] = 'Strong Buy'
            confidence_scores.append(1 - (rsi/30))
        elif rsi > 70:
            signals['RSI'] = 'Strong Sell'
            confidence_scores.append((rsi-70)/30)
        else:
            signals['RSI'] = 'Neutral'
            confidence_scores.append(0.5)
        
        # MACD Signals
        macd = df['macd'].iloc[-1]
        macd_prev = df['macd'].iloc[-2]
        if macd > 0 and macd_prev < 0:
            signals['MACD'] = 'Strong Buy'
            confidence_scores.append(0.8)
        elif macd < 0 and macd_prev > 0:
            signals['MACD'] = 'Strong Sell'
            confidence_scores.append(0.8)
        else:
            signals['MACD'] = 'Hold'
            confidence_scores.append(0.5)
        
        # Moving Average Signals
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        price = df['Close'].iloc[-1]
        ma_confidence = abs((sma_20 - sma_50)/sma_50)
        if sma_20 > sma_50:
            signals['MA'] = 'Bullish'
            confidence_scores.append(min(ma_confidence, 1.0))
        else:
            signals['MA'] = 'Bearish'
            confidence_scores.append(min(ma_confidence, 1.0))
        
        # Bollinger Bands Signals
        bb_upper = df['bbands_upper'].iloc[-1]
        bb_lower = df['bbands_lower'].iloc[-1]
        bb_confidence = abs((price - bb_lower)/(bb_upper - bb_lower))
        if price < bb_lower:
            signals['BB'] = 'Strong Buy'
            confidence_scores.append(1 - bb_confidence)
        elif price > bb_upper:
            signals['BB'] = 'Strong Sell'
            confidence_scores.append(bb_confidence)
        else:
            signals['BB'] = 'Neutral'
            confidence_scores.append(0.5)
        
        # Pattern Recognition Signals
        if df['doji'].iloc[-1]:
            signals['Pattern'] = 'Doji - Potential Reversal'
            confidence_scores.append(0.6)
        elif df['hammer'].iloc[-1]:
            signals['Pattern'] = 'Hammer - Potential Bullish'
            confidence_scores.append(0.7)
        elif df['engulfing'].iloc[-1]:
            signals['Pattern'] = 'Engulfing - Strong Signal'
            confidence_scores.append(0.8)
        elif df['morning_star'].iloc[-1]:
            signals['Pattern'] = 'Morning Star - Very Bullish'
            confidence_scores.append(0.9)
        
        # Calculate overall confidence
        signals['overall_confidence'] = np.mean(confidence_scores)
        
        return signals

class NewsAnalyzer:
    def __init__(self):
        pass
    
    def analyze_news(self, news_items: List[Dict]) -> Dict:
        """Analyze news sentiment using TextBlob instead of transformers"""
        sentiments = []
        for item in news_items:
            try:
                # Get published date from various possible keys
                published_date = None
                if 'published' in item:
                    published_date = item['published']
                elif 'providerPublishTime' in item:
                    published_date = datetime.fromtimestamp(item['providerPublishTime'])
                else:
                    published_date = datetime.now()  # Fallback if no date found

                # Get title and content
                title = item.get('title', '')
                content = item.get('summary', '') or item.get('text', '') or ''
                
                # Analyze title sentiment
                title_sentiment = TextBlob(title).sentiment.polarity if title else 0
                
                # Analyze content sentiment if available
                content_sentiment = TextBlob(content).sentiment.polarity if content else title_sentiment
                
                # Combine sentiments with weighted average
                combined_sentiment = (title_sentiment * 0.6 + content_sentiment * 0.4)
                
                sentiments.append({
                    'title': title,
                    'sentiment': combined_sentiment,
                    'date': published_date
                })
                
            except Exception as e:
                logger.error(f"Error processing news item: {e}")
                continue
        
        # Calculate aggregate sentiment
        if sentiments:
            recent_sentiment = np.mean([s['sentiment'] for s in sentiments[:5]])
            overall_sentiment = np.mean([s['sentiment'] for s in sentiments])
            
            return {
                'recent_sentiment': recent_sentiment,
                'overall_sentiment': overall_sentiment,
                'sentiment_trend': 'Improving' if recent_sentiment > overall_sentiment else 'Declining',
                'details': sentiments
            }
        
        return {
            'recent_sentiment': 0,
            'overall_sentiment': 0,
            'sentiment_trend': 'Neutral',
            'details': []
        }

class PricePredictionModel:
    def __init__(self, input_dim=5, hidden_dim=32, num_layers=2, output_dim=1):
        self.model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for LSTM model"""
        # Select features for prediction
        features = ['Close', 'Volume', 'rsi', 'macd', 'obv']
        data = df[features].values
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])  # Predict close price
        
        return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

    def train(self, df: pd.DataFrame):
        """Train the model on historical data"""
        X, y = self.prepare_data(df)
        
        # Train model
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        self.model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    def predict(self, df: pd.DataFrame, days_ahead: int = 5) -> List[float]:
        """Make price predictions"""
        self.model.eval()
        
        # Prepare last sequence
        X, _ = self.prepare_data(df)
        last_sequence = X[-1].unsqueeze(0)
        
        predictions = []
        for _ in range(days_ahead):
            with torch.no_grad():
                pred = self.model(last_sequence)
                predictions.append(pred.item())
                
                # Update sequence for next prediction
                last_sequence = torch.cat((last_sequence[:, 1:, :], pred.unsqueeze(0).unsqueeze(0)), dim=1)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()

class TradingBot:
    def __init__(self):
        self.technical_analysis = TechnicalAnalysis()
        self.news_analyzer = NewsAnalyzer()
        self.price_predictor = PricePredictionModel()
            
    def analyze_stock(self, symbol: str, df: pd.DataFrame, news: List[Dict]) -> Dict:
        """Perform comprehensive stock analysis"""
        try:
            # Calculate technical indicators
            df = self.technical_analysis.calculate_indicators(df)
            technical_signals = self.technical_analysis.generate_signals(df)
                
            # Analyze news sentiment
            news_analysis = self.news_analyzer.analyze_news(news)
                
            # Generate price predictions
            try:
                self.price_predictor.train(df)
                price_predictions = self.price_predictor.predict(df)
            except Exception as e:
                logger.error(f"Error in price prediction: {e}")
                price_predictions = []
                
            # Generate recommendation
                recommendation = self._generate_recommendation(
                technical_signals, 
                news_analysis, 
                price_predictions
            )
                
            return {
                'technical_signals': technical_signals,
                'news_sentiment': news_analysis,
                'price_predictions': price_predictions,
                'overall_confidence': technical_signals.get('overall_confidence', 0),
                'recommendation': recommendation
            }
                
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {e}")
            return {
                'technical_signals': {},
                'news_sentiment': {
                    'recent_sentiment': 0,
                    'overall_sentiment': 0,
                    'sentiment_trend': 'Neutral',
                    'details': []
                },
                'price_predictions': [],
                'overall_confidence': 0,
                'recommendation': {
                    'action': 'HOLD',
                    'confidence': 0,
                    'reasons': f'Error analyzing stock: {str(e)}'
                }
            }
    
    def _generate_recommendation(self, technical_signals: Dict, news_analysis: Dict, price_predictions: List) -> Dict:
        """Generate trading recommendation based on all signals"""
        buy_signals = 0
        sell_signals = 0
        
        # Technical Analysis Contribution
        if technical_signals.get('RSI') == 'Strong Buy': buy_signals += 2
        elif technical_signals.get('RSI') == 'Strong Sell': sell_signals += 2
        
        if technical_signals.get('MACD') == 'Strong Buy': buy_signals += 2
        elif technical_signals.get('MACD') == 'Strong Sell': sell_signals += 2
        
        if technical_signals.get('MA') == 'Bullish': buy_signals += 1
        elif technical_signals.get('MA') == 'Bearish': sell_signals += 1
        
        # News Sentiment Contribution
        if news_analysis['recent_sentiment'] > 0.2: buy_signals += 2
        elif news_analysis['recent_sentiment'] < -0.2: sell_signals += 2
        
        # Price Prediction Contribution
        if len(price_predictions) > 0:
            current_price = price_predictions[0]
            future_price = price_predictions[-1]
            if future_price > current_price * 1.02: buy_signals += 2
            elif future_price < current_price * 0.98: sell_signals += 2
        
        # Generate Final Recommendation
        total_signals = max(buy_signals + sell_signals, 1)  # Avoid division by zero
        buy_confidence = buy_signals / total_signals
        sell_confidence = sell_signals / total_signals
        
        if buy_confidence > sell_confidence and buy_confidence > 0.6:
            action = 'BUY'
            confidence = buy_confidence
        elif sell_confidence > buy_confidence and sell_confidence > 0.6:
            action = 'SELL'
            confidence = sell_confidence
        else:
            action = 'HOLD'
            confidence = max(buy_confidence, sell_confidence)
        
        return {
            'action': action,
            'confidence': confidence,
            'reasons': self._generate_reason_text(technical_signals, news_analysis, price_predictions)
        }
    
    def _generate_reason_text(self, technical_signals: Dict, news_analysis: Dict, price_predictions: List) -> str:
        """Generate detailed explanation for the recommendation"""
        reasons = []
        
        # Technical Analysis Summary
        tech_bullish = sum(1 for signal in technical_signals.values() if 'Buy' in str(signal) or 'Bullish' in str(signal))
        tech_bearish = sum(1 for signal in technical_signals.values() if 'Sell' in str(signal) or 'Bearish' in str(signal))
        
        if tech_bullish > tech_bearish:
            reasons.append(f"Technical indicators are bullish ({tech_bullish} positive signals)")
        elif tech_bearish > tech_bullish:
            reasons.append(f"Technical indicators are bearish ({tech_bearish} negative signals)")
        
        # News Sentiment Summary
        if news_analysis['recent_sentiment'] > 0.2:
            reasons.append("Recent news sentiment is positive")
        elif news_analysis['recent_sentiment'] < -0.2:
            reasons.append("Recent news sentiment is negative")
        
        # Price Prediction Summary
        if len(price_predictions) > 0:
            price_change = ((price_predictions[-1] - price_predictions[0]) / price_predictions[0]) * 100
            reasons.append(f"Predicted price change: {price_change:.2f}% over next {len(price_predictions)} days")
        
        return " | ".join(reasons)

class StockTracker:
    def __init__(self, db_path: str = "stock_tracker.db"):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()
        self.trading_bot = TradingBot()
        
    def setup_database(self):
        """Initialize database tables"""
        cursor = self.conn.cursor()
        
        # Watched Stocks Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watched_stocks (
                symbol TEXT PRIMARY KEY,
                added_date TIMESTAMP,
                last_analysis TIMESTAMP,
                last_recommendation TEXT
            )
        ''')
        
        # Trading History Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_history (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                date TIMESTAMP,
                action TEXT,
                price FLOAT,
                quantity INTEGER,
                confidence FLOAT,
                reasons TEXT
            )
        ''')
        
        # Analysis History Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                date TIMESTAMP,
                technical_signals TEXT,
                news_sentiment FLOAT,
                prediction_accuracy FLOAT
            )
        ''')
        
        self.conn.commit()

    def add_stock(self, symbol: str) -> bool:
        """Add a stock to watchlist"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO watched_stocks (symbol, added_date) VALUES (?, ?)",
                (symbol.upper(), datetime.now())
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def remove_stock(self, symbol: str) -> bool:
        """Remove a stock from watchlist"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM watched_stocks WHERE symbol = ?", (symbol.upper(),))
        self.conn.commit()
        return True
    
    def get_watched_stocks(self) -> List[str]:
        """Get list of watched stocks"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT symbol FROM watched_stocks")
        return [row[0] for row in cursor.fetchall()]
    
    def analyze_stock(self, symbol: str) -> Dict:
        """Perform comprehensive stock analysis"""
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Get news with error handling
            try:
                news = stock.news
            except Exception as e:
                logger.warning(f"Error fetching news for {symbol}: {e}")
                news = []
            
            # Perform analysis
            analysis = self.trading_bot.analyze_stock(symbol, df, news)
            
            # Store analysis results
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE watched_stocks 
                SET last_analysis = ?, last_recommendation = ?
                WHERE symbol = ?
            ''', (
                datetime.now(),
                json.dumps(analysis['recommendation']),
                symbol
            ))
            
            cursor.execute('''
                INSERT INTO analysis_history 
                (symbol, date, technical_signals, news_sentiment, prediction_accuracy)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now(),
                json.dumps(analysis['technical_signals']),
                analysis['news_sentiment']['recent_sentiment'],
                0.0  # Will be updated when we can measure accuracy
            ))
            
            self.conn.commit()
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {e}")
            return {
                'technical_signals': {},
                'news_sentiment': {
                    'recent_sentiment': 0,
                    'overall_sentiment': 0,
                    'sentiment_trend': 'Neutral',
                    'details': []
                },
                'price_predictions': [],
                'overall_confidence': 0,
                'recommendation': {
                    'action': 'HOLD',
                    'confidence': 0,
                    'reasons': f'Error analyzing stock: {str(e)}'
                }
            }

def run_streamlit_app():
    st.set_page_config(page_title="Advanced Trading Bot", layout="wide")
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .tradingCard {
            background-color: #1e2127;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .confidence-high { color: #00ff00; }
        .confidence-medium { color: #ffff00; }
        .confidence-low { color: #ff0000; }
        </style>
    """, unsafe_allow_html=True)
    
    tracker = StockTracker()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Trading Bot")
        
        # Add/Remove Stocks
        st.header("Manage Stocks")
        new_stock = st.text_input("Add Stock Symbol").upper()
        if st.button("Add Stock"):
            if new_stock:
                if tracker.add_stock(new_stock):
                    st.success(f"Added {new_stock} to watchlist")
                else:
                    st.error("Stock already in watchlist")
        
        watched_stocks = tracker.get_watched_stocks()
        if watched_stocks:
            stock_to_remove = st.selectbox("Remove Stock", [""] + watched_stocks)
            if st.button("Remove Selected Stock") and stock_to_remove:
                tracker.remove_stock(stock_to_remove)
                st.success(f"Removed {stock_to_remove}")
                st.rerun()
    
    # Main Content
    st.title("📊 Advanced Trading Bot Dashboard")
    
    # Stock Analysis Section
    if not watched_stocks:
        st.warning("Please add stocks to your watchlist to begin analysis")
    else:
        tabs = st.tabs(["Active Signals", "Technical Analysis", "News Impact", "Performance"])
        
        # Active Signals Tab
        with tabs[0]:
            st.header("Active Trading Signals")
            
            for symbol in watched_stocks:
                analysis = tracker.analyze_stock(symbol)
                recommendation = analysis['recommendation']
                
                # Create expandable card for each stock
                with st.expander(f"{symbol} - {recommendation['action']}", expanded=True):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.subheader("Signal Strength")
                        confidence_color = (
                            "confidence-high" if recommendation['confidence'] > 0.7
                            else "confidence-medium" if recommendation['confidence'] > 0.5
                            else "confidence-low"
                        )
                        st.markdown(f"<h3 class='{confidence_color}'>{recommendation['confidence']:.2%}</h3>",
                                  unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("Reasoning")
                        st.write(recommendation['reasons'])
                    
                    with col3:
                        st.subheader("Action")
                        if recommendation['action'] == 'BUY':
                            st.success("BUY")
                        elif recommendation['action'] == 'SELL':
                            st.error("SELL")
                        else:
                            st.info("HOLD")
                    
                    # Technical Indicators Summary
                    st.subheader("Technical Indicators")
                    indicators = analysis['technical_signals']
                    cols = st.columns(4)
                    for i, (indicator, value) in enumerate(indicators.items()):
                        if indicator != 'overall_confidence':
                            cols[i % 4].metric(indicator, value)
                    
                    # Price Predictions
                    if analysis['price_predictions']:
                        st.subheader("Price Predictions")
                        pred_df = pd.DataFrame({
                            'Day': range(1, len(analysis['price_predictions']) + 1),
                            'Predicted Price': analysis['price_predictions']
                        })
                        fig = px.line(pred_df, x='Day', y='Predicted Price',
                                    title=f"{symbol} Price Prediction - Next {len(analysis['price_predictions'])} Days")
                        st.plotly_chart(fig, use_container_width=True)

        # Other tabs implementation continues...
        # Technical Analysis Tab
        with tabs[1]:
            st.header("Technical Analysis Dashboard")
            selected_stock = st.selectbox("Select Stock for Analysis", watched_stocks, key="tech_analysis")
            
            if selected_stock:
                # Fetch data
                stock = yf.Ticker(selected_stock)
                df = stock.history(period="1y")
                df = tracker.trading_bot.technical_analysis.calculate_indicators(df)
                
                # Main Price Chart
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ))
                
                # Add moving averages
                fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='blue')))
                
                # Add Bollinger Bands
                fig.add_trace(go.Scatter(x=df.index, y=df['bbands_upper'], name='BB Upper',
                                       line=dict(color='gray', dash='dash')))
                fig.add_trace(go.Scatter(x=df.index, y=df['bbands_lower'], name='BB Lower',
                                       line=dict(color='gray', dash='dash')))
                
                fig.update_layout(
                    title=f"{selected_stock} Price and Indicators",
                    yaxis_title="Price",
                    height=600,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical Indicators Grid
                st.subheader("Technical Indicators")
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI Chart
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI'))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(title="RSI", height=300, template="plotly_dark")
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # MACD Chart
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD'))
                    fig_macd.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_macd.update_layout(title="MACD", height=300, template="plotly_dark")
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                with col2:
                    # Volume Chart
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
                    fig_vol.update_layout(title="Volume", height=300, template="plotly_dark")
                    st.plotly_chart(fig_vol, use_container_width=True)
                    
                    # Money Flow Index
                    fig_mfi = go.Figure()
                    fig_mfi.add_trace(go.Scatter(x=df.index, y=df['mfi'], name='MFI'))
                    fig_mfi.add_hline(y=80, line_dash="dash", line_color="red")
                    fig_mfi.add_hline(y=20, line_dash="dash", line_color="green")
                    fig_mfi.update_layout(title="Money Flow Index", height=300, template="plotly_dark")
                    st.plotly_chart(fig_mfi, use_container_width=True)

        # News Impact Tab
        with tabs[2]:
            st.header("News Sentiment Analysis")
            selected_stock = st.selectbox("Select Stock for News Analysis", watched_stocks, key="news_analysis")
            
            if selected_stock:
                stock = yf.Ticker(selected_stock)
                news_items = stock.news
                news_analysis = tracker.trading_bot.news_analyzer.analyze_news(news_items)
                
                # Sentiment Overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    sentiment_score = news_analysis['recent_sentiment']
                    st.metric("Recent Sentiment", 
                             f"{sentiment_score:.2f}",
                             delta=f"{sentiment_score - news_analysis['overall_sentiment']:.2f}")
                
                with col2:
                    st.metric("Overall Sentiment", f"{news_analysis['overall_sentiment']:.2f}")
                
                with col3:
                    st.metric("Sentiment Trend", news_analysis['sentiment_trend'])
                
                # News Timeline
                st.subheader("Recent News Analysis")
                for news_item in news_analysis['details']:
                    sentiment_color = (
                        "🟢" if news_item['sentiment'] > 0.2
                        else "🔴" if news_item['sentiment'] < -0.2
                        else "⚪"
                    )
                    
                    with st.expander(f"{sentiment_color} {news_item['title']}"):
                        st.write(f"Date: {news_item['date']}")
                        st.write(f"Sentiment Score: {news_item['sentiment']:.2f}")
                        if 'summary' in news_item:
                            st.write(news_item['summary'])

        # Performance Tab
        with tabs[3]:
            st.header("Bot Performance Metrics")
            
            # Date Range Selector
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())
            
            # Get performance data from database
            cursor = tracker.conn.cursor()
            cursor.execute("""
                SELECT symbol, date, technical_signals, news_sentiment, prediction_accuracy 
                FROM analysis_history 
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """, (start_date, end_date))
            
            performance_data = cursor.fetchall()
            
            if performance_data:
                # Convert to DataFrame
                perf_df = pd.DataFrame(performance_data, 
                                     columns=['Symbol', 'Date', 'Technical_Signals', 
                                            'News_Sentiment', 'Prediction_Accuracy'])
                
                # Accuracy Metrics
                st.subheader("Prediction Accuracy")
                fig_accuracy = px.line(perf_df, x='Date', y='Prediction_Accuracy', 
                                     color='Symbol', title="Model Accuracy Over Time")
                st.plotly_chart(fig_accuracy, use_container_width=True)
                
                # Sentiment vs Technical Signals
                st.subheader("Signal Analysis")
                for symbol in perf_df['Symbol'].unique():
                    symbol_data = perf_df[perf_df['Symbol'] == symbol]
                    with st.expander(f"{symbol} Analysis"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_sentiment = px.line(symbol_data, x='Date', y='News_Sentiment',
                                                  title="News Sentiment Trend")
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        with col2:
                            # Parse technical signals from JSON
                            signals_df = pd.DataFrame([
                                json.loads(signals) for signals in symbol_data['Technical_Signals']
                            ])
                            if not signals_df.empty:
                                fig_signals = px.line(signals_df, title="Technical Signals Strength")
                                st.plotly_chart(fig_signals, use_container_width=True)
            else:
                st.info("No performance data available for the selected date range")

if __name__ == "__main__":
    run_streamlit_app()