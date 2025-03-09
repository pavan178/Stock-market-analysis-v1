from flask import Flask, render_template, request, jsonify, session
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import nltk
import time
from flask_caching import Cache
import random
import re
from model import predict_future_prices
nltk.download('vader_lexicon')
print(nltk.__version__) 
# Ensure required NLTK data is downloaded
nltk.download('vader_lexicon')

app = Flask(__name__)
sid = SentimentIntensityAnalyzer()

# Configure cache
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 minutes
cache = Cache(app)

# --- Data Fetching and Enrichment ---
@cache.memoize(300)
def fetch_stock_data(ticker):
    """Fetch stock data with proper symbol validation"""
    try:
        # Convert to official symbol format
        ticker = ticker.upper().strip()
        
        # Improved regex validation (allows numbers and some special characters)
        if not re.match(r'^[A-Z0-9]{1,5}(\.?[A-Z]{1,2})?$', ticker):
            return pd.DataFrame(), {}, 'Unknown'
            
        # Add request throttling
        time.sleep(0.2)  # Prevent rate limiting
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # More specific validation
        if not info:
            return pd.DataFrame(), {}, 'Unknown'
        elif 'currentPrice' not in info and 'regularMarketPrice' not in info:
            return pd.DataFrame(), {}, info.get('sector', 'Unknown')
            
        return stock.history(period="5y"), info, info.get('sector', 'Unknown')
        
    except Exception as e:
        error_type = type(e).__name__
        print(f"Error fetching {ticker}: {error_type} - {str(e)}")
        return pd.DataFrame(), {}, 'Unknown'

# --- Enhanced Technical Analysis ---
def calculate_technical_indicators(df):
    """Calculate technical indicators with improved error handling"""
    if df.empty or len(df) < 50:  # Need enough data for indicators
        return df
        
    # Clean data first
    df = df.copy()
    df.fillna(method='ffill', inplace=True)
    
    # Basic indicators
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # Handle volume trends
    df['VolumeMA20'] = df['Volume'].rolling(window=20).mean()
    df['VolumeTrend'] = df['Volume'] / df['VolumeMA20']
    
    try:
        # Calculate RSI safely
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # Safe MACD calculation
        macd = ta.macd(df['Close'])
        if isinstance(macd, pd.DataFrame) and not macd.empty:
            # Handle different column naming patterns
            macd_cols = [col for col in macd.columns if 'MACD_' in col]
            signal_cols = [col for col in macd.columns if 'MACD' in col and 's_' in col.lower()]
            
            if macd_cols and signal_cols:
                df['MACD'] = macd[macd_cols[0]]
                df['Signal'] = macd[signal_cols[0]]
    except Exception as e:
        print(f"Error calculating momentum indicators: {str(e)}")
    
    try:
        # Bollinger Bands with safe extraction
        bands = ta.bbands(df['Close'], length=20, std=2)
        if isinstance(bands, pd.DataFrame) and not bands.empty:
            band_columns = bands.columns.tolist()
            lower_cols = [col for col in band_columns if 'BBL_' in col]
            middle_cols = [col for col in band_columns if 'BBM_' in col]
            upper_cols = [col for col in band_columns if 'BBU_' in col]
            
            if lower_cols and middle_cols and upper_cols:
                df['LowerBand'] = bands[lower_cols[0]]
                df['MiddleBand'] = bands[middle_cols[0]]
                df['UpperBand'] = bands[upper_cols[0]]
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {str(e)}")
        
    # Fill NaN values that might have been introduced
    df.fillna(method='bfill', inplace=True)
    
    return df



# --- Extended Sentiment Analysis ---
def analyze_sentiment(text):
    """Enhanced sentiment analysis with validation"""
    try:
        if not text.strip():
            return 0
            
        scores = sid.polarity_scores(text)
        print(f"Sentiment analysis results: {scores}")  # Debug output
        if abs(scores['compound']) < 0.05:  # Neutral threshold
            print("Neutral sentiment detected")
        return scores['compound']
    except Exception as e:
        print(f"Sentiment analysis error: {str(e)}")
        return 0

# --- Composite Scoring for Buying Suggestions ---
def compute_composite_score(df, info, sentiment_score, ticker):
    try:
        # Add 52-week calculations
        df['52_week_high'] = df['High'].rolling(252).max()
        df['52_week_low'] = df['Low'].rolling(252).min()
        latest = df.iloc[-1]
        
        # Add moving average interpretation
        ma_50 = latest.get('SMA50', 0)
        ma_200 = df['Close'].rolling(window=200).mean().iloc[-1]
        price_vs_50 = latest['Close'] > ma_50
        price_vs_200 = latest['Close'] > ma_200
        
        # Update technical score calculation
        ma_score = 0.2 if price_vs_50 else -0.2
        ma_score += 0.3 if price_vs_200 else -0.3
        
        # Update technical component
        rsi = latest.get('RSI', 50)
        tech_score = max(0, (70 - rsi) / 40) + ma_score
        
        # Fundamental Component
        pe = info.get('trailingPE', 20)  # Default to industry average
        fundamental_score = max(0, (20 - pe) / 20) if pe else 0
        
        # Sentiment Component
        sentiment_component = (sentiment_score + 1) / 2
        
        # Volume Analysis
        volume_score = 0
        if latest.get('VolumeMA20', 0) > 0:
            volume_score = min(1, latest['Volume'] / latest['VolumeMA20'])
            
        # Momentum Analysis
        momentum_score = 1 if (
            latest.get('Close', 0) > latest.get('ISA_9', 0) and 
            latest.get('Close', 0) > latest.get('ISB_26', 0)
        ) else 0

        # Fair Value Relationship
        current_price = latest['Close']
        fair_price = fair_price_estimate(info) or current_price  # Prevent None
        
        
        # Calculate percentage difference from fair price
        if fair_price > 0:
            price_to_fair = current_price / fair_price
            price_diff_pct = abs(1 - price_to_fair) * 100
            
            # Determine valuation status with specific percentages
            if price_to_fair < 0.9:  # More than 10% undervalued
                valuation_status = "Significantly Undervalued"
            elif price_to_fair < 0.97:  # 3-10% undervalued
                valuation_status = "Moderately Undervalued"
            elif price_to_fair <= 1.03:  # Within 3% of fair value
                valuation_status = "Fairly Valued"
            elif price_to_fair <= 1.1:  # 3-10% overvalued
                valuation_status = "Moderately Overvalued"
            else:  # More than 10% overvalued
                valuation_status = "Significantly Overvalued"
        else:
            price_to_fair = 1
            price_diff_pct = 0
            valuation_status = "Fair Value Unknown"
        
        # Valuation Impact (0-1 scale)
        # Create a curve that gives higher scores when undervalued
        if price_to_fair < 1:  # Undervalued
            valuation_impact = min(1, 1.5 * (1 - price_to_fair))  # Boost undervalued stocks
        else:  # Overvalued
            valuation_impact = max(0, 1 - (price_to_fair - 1) * 2)  # Penalize overvalued stocks more
        
        # Update Composite Formula
        composite = (
            0.25 * tech_score +          # Technicals
            0.3 * valuation_impact +     # Valuation (most weight)
            0.2 * sentiment_component +  # Sentiment
            0.15 * volume_score +        # Volume
            0.1 * momentum_score         # Momentum
        )
        
        # Generate Correlation Explanation
        explanation = f"""
        <div class="analysis-header">{ticker or 'Unknown Ticker'} Analysis</div>
        <div class="analysis-details">
            <div class="analysis-row">
                <span class="label">Technical Indicators:</span>
                <span class="value {'good' if tech_score > 0.5 else 'bad'}">
                    {'Bullish' if tech_score > 0.5 else 'Bearish'} (RSI: {rsi:.1f})
                </span>
            </div>
            <div class="analysis-row">
                <span class="label">Valuation:</span>
                <span class="value {'good' if valuation_impact > 0.5 else 'bad'}">
                    {valuation_status}
                </span>
            </div>
            <div class="analysis-row">
                <span class="label">Market Sentiment:</span>
                <span class="value {'good' if sentiment_score > 0 else 'bad'}">
                    {'Positive' if sentiment_score > 0 else 'Negative'}
                </span>
            </div>
            <div class="analysis-row">
                <span class="label">Volume Trend:</span>
                <span class="value {'good' if volume_score > 1 else 'bad'}">
                    {'Above average' if volume_score > 1 else 'Below average'}
                </span>
            </div>
            <div class="analysis-row">
                <span class="label">Momentum:</span>
                <span class="value {'good' if momentum_score == 1 else 'neutral'}">
                    {'Bullish' if momentum_score == 1 else 'Neutral'}
                </span>
            </div>
            <div class="analysis-row">
                <span class="label">Moving Averages:</span>
                <span class="value {'good' if price_vs_50 else 'bad'}">
                    Price {'above' if price_vs_50 else 'below'} 50-day MA (${ma_50:.2f})
                </span>
                <span class="value {'good' if price_vs_200 else 'bad'}">
                    Price {'above' if price_vs_200 else 'below'} 200-day MA (${ma_200:.2f})
                </span>
            </div>
        </div>
        <div class="correlation">
            <div class="fair-value">
                Current Price: ${current_price:.2f} vs 
                Fair Value: ${fair_price:.2f} ({valuation_status} by {price_diff_pct:.1f}%)
            </div>
            <div class="weighting">
                Recommendation Weighting:<br>
                - Valuation: 30%<br>
                - Technicals: 25%<br>
                - Sentiment: 20%<br>
                - Volume: 15%<br>
                - Momentum: 10%
            </div>
        </div>
        """

        # Return 52-week data
        return composite, explanation, {
            '52_week_high': latest['52_week_high'],
            '52_week_low': latest['52_week_low'],
            'current_price': latest['Close'],
            'fair_price': fair_price,
            'price_diff_pct': price_diff_pct,
            'valuation_status': valuation_status
        }
    except Exception as e:
        print(f"Error generating composite score: {str(e)}")
        return 0, "Unable to generate analysis", {}




def fair_price_estimate(info):
    """
    Calculates a fair price estimate based on multiple valuation methods.
    
    Args:
        info (dict): Stock information dictionary from yfinance
        
    Returns:
        float or None: Estimated fair price or None if not enough data
    """
    try:
        # Get basic financial metrics
        pe = info.get('trailingPE')
        eps = info.get('trailingEps')
        forward_pe = info.get('forwardPE')
        forward_eps = info.get('forwardEps')
        pb = info.get('priceToBook')
        sector = info.get('sector', 'Unknown')
        
        # Set sector-specific PE premium/discount factors
        sector_adjustments = {
            'Technology': 1.15,  # High growth potential
            'Healthcare': 1.10,  # Defensive with growth
            'Financial': 0.90,   # Usually trade at discount to market
            'Energy': 0.80,      # Cyclical sector
            'Consumer Cyclical': 0.95,
            'Consumer Defensive': 1.05,
            'Utilities': 0.85,
            'Communication Services': 1.05,
            'Industrial': 1.00,
            'Basic Materials': 0.90,
            'Real Estate': 0.95
        }
        
        # Default adjustment factor if sector not found
        sector_factor = sector_adjustments.get(sector, 1.0)
        
        # Start with empty estimates list and weights
        estimates = []
        weights = []
        
        # 1. PE x EPS method (trailing)
        if pe and eps and pe > 0 and eps > 0:
            pe_estimate = pe * eps * sector_factor
            # Higher quality estimate gets higher weight
            estimates.append(pe_estimate)
            weights.append(3)
        
        # 2. Forward PE x Forward EPS (if available)
        if forward_pe and forward_eps and forward_pe > 0 and forward_eps > 0:
            forward_estimate = forward_pe * forward_eps * sector_factor
            # Forward estimates often more relevant
            estimates.append(forward_estimate)
            weights.append(4)
        
        # 3. Book value method (if available)
        if pb and pb > 0 and info.get('bookValue'):
            book_estimate = pb * info.get('bookValue') * sector_factor
            estimates.append(book_estimate)
            weights.append(2)
        
        # 4. Dividend discount model (if pays dividend)
        if info.get('dividendYield') and info.get('dividendRate'):
            # Simple dividend discount model
            div_yield = info.get('dividendYield')
            if div_yield > 0:
                # Assuming 3% long-term growth
                growth_rate = 0.03
                # Required rate of return based on sector
                required_return = {
                    'Technology': 0.10,
                    'Healthcare': 0.09,
                    'Financial': 0.08,
                    'Energy': 0.09,
                    'Utilities': 0.07
                }.get(sector, 0.08)
                
                if required_return > growth_rate:
                    div_estimate = info.get('dividendRate') * (1 + growth_rate) / (required_return - growth_rate)
                    estimates.append(div_estimate)
                    weights.append(1)  # Lower weight for dividend model
        
        # Calculate weighted average if we have estimates
        if estimates:
            fair_price = sum(est * weight for est, weight in zip(estimates, weights)) / sum(weights)
            return fair_price
        else:
            # Fallback to current price if no metrics available
            return info.get('currentPrice') or info.get('regularMarketPrice')
            
    except Exception as e:
        print(f"Error calculating fair value: {str(e)}")
        return None

def get_buying_recommendation(composite_score, sector):
    # Sector-specific thresholds
    thresholds = {
        'Technology': (0.8, 0.6, 0.4),
        'Healthcare': (0.7, 0.5, 0.3),
        'Financial': (0.6, 0.4, 0.2)
    }
    
    strong, buy, hold = thresholds.get(sector, (0.7, 0.5, 0.3))
    
    if composite_score >= strong:
        return "Strong Buy"
    elif composite_score >= buy:
        return "Buy" 
    elif composite_score >= hold:
        return "Hold"
    else:
        return "Sell"

# --- API Endpoints ---
@app.route('/')
def index():
    # For demonstration, list popular stocks with sectors
    stocks = [
        {"ticker": "AAPL", "sector": "Technology"},
        {"ticker": "GOOGL", "sector": "Technology"},
        {"ticker": "MSFT", "sector": "Technology"},
        {"ticker": "JPM", "sector": "Financial"},
        {"ticker": "PFE", "sector": "Healthcare"}
    ]
    return render_template("index.html", stocks=stocks)
@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form.get('ticker', '').upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400
    
    try:
        df, info, sector = fetch_stock_data(ticker)
        if df.empty:
            # Check if ticker might be invalid
            valid_tickers = [stock['symbol'] for stock in get_tickers().json]
            close_matches = [t for t in valid_tickers if t.startswith(ticker[:3])]
            
            error_msg = f"No data found for ticker: {ticker}"
            if close_matches:
                error_msg += f". Did you mean: {', '.join(close_matches[:3])}?"
            
            return jsonify({"error": error_msg}), 404
        
        df = calculate_technical_indicators(df)
        
        # Get sentiment score AND headlines
        sentiment_score, headlines = get_actual_sentiment(ticker)  # Modified to return tuple
        
        # Pass all required parameters including ticker
        composite_score, explanation, data = compute_composite_score(df, info, sentiment_score, ticker)
        
        recommendation = get_buying_recommendation(composite_score, sector)
        fair_price = fair_price_estimate(info)
        latest_price = df['Close'].iloc[-1]
        
        # Add prediction data
        prediction = predict_future_prices(ticker)
        
        response = {
            "ticker": ticker,
            "sector": sector,
            "latest_price": latest_price,
            "recommendation": recommendation,
            "composite_score": composite_score,
            "fair_price": fair_price if fair_price else "N/A",
            "sentiment_score": sentiment_score,
            "sentiment_details": {
                "headline_count": len(headlines),
                "sample_headlines": headlines[:2],
                "raw_score": sentiment_score
            },
            "explanation": explanation,
            "52_week_data": data,
            "prediction": prediction
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/tickers')
@cache.cached(timeout=3600)  # Cache for 1 hour
def get_tickers():
    try:
        # Get S&P 500 components from Wikipedia
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500 = table[0]
        return jsonify([{
            "symbol": row['Symbol'],
            "name": row['Security'],
            "sector": row['GICS Sector']
        } for _, row in sp500.iterrows()])
    except Exception as e:
        return jsonify([])

@app.route('/realtime/<ticker>')
def realtime_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d', interval='1m')
    latest = data.iloc[-1]
    return jsonify({
        "price": latest['Close'],
        "change": (latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100,
        "volume": latest['Volume']
    })

def get_actual_sentiment(ticker):
    """Fetch real news sentiment with current Yahoo Finance structure handling"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []
        
        if not news:
            print(f"No news found for {ticker}")
            return 0, []
            
        headlines = []
        for idx, item in enumerate(news):
            try:
                # Extract title from current structure
                title = (
                    item.get('title') or 
                    item.get('content', {}).get('title') or 
                    item.get('link', '').split('/')[-1].replace('-', ' ')
                )
                
                # Extract publisher from current structure
                publisher = (
                    item.get('provider') or 
                    item.get('publisher') or 
                    item.get('source', 'Unknown')
                )
                
                # Clean up HTML entities and special characters
                title = (
                    title.replace('&#39;', "'")
                    .replace('&amp;', '&')
                    .encode('utf-8', 'ignore')
                    .decode('utf-8')
                    .strip()
                )
                
                # Basic validation with content check
                if len(title) > 5 and any(c.isalpha() for c in title):
                    headlines.append(f"{publisher}: {title}")
                    
            except Exception as item_error:
                print(f"Error processing item {idx}: {str(item_error)[:50]}")
                continue

        if not headlines:
            print(f"No valid headlines for {ticker} (sample item keys: {list(news[0].keys())})")
            return 0, []
            
        print(f"Processing {len(headlines)} headlines for {ticker}")
        return analyze_sentiment(" ".join(headlines)), headlines  # Return tuple
        
    except Exception as e:
        print(f"Critical sentiment error: {str(e)}")
        return 0, []  # Return default tuple

@app.route('/debug/<ticker>')
def debug_news(ticker):
    news = yf.Ticker(ticker).news or []
    return jsonify({
        "news_count": len(news),
        "sample_items": news[:2] if news else [],
        "status": "success" if news else "no news"
    })

@app.route('/market-indices')
def get_market_indices():
    indices = {
        'SP500': yf.Ticker("^GSPC").history(period='1d').iloc[-1].to_dict(),
        'NASDAQ': yf.Ticker("^IXIC").history(period='1d').iloc[-1].to_dict(),
        'DOW': yf.Ticker("^DJI").history(period='1d').iloc[-1].to_dict()
    }
    return jsonify(indices)

@app.route('/watchlist', methods=['GET', 'POST'])
def manage_watchlist():
    if request.method == 'POST':
        try:
            data = request.get_json()
            action = data.get('action')
            symbol = data.get('symbol')
            
            # Get current watchlist from session
            watchlist = session.get('watchlist', [])
            
            if action == 'add' and symbol not in watchlist:
                watchlist.append(symbol)
            elif action == 'remove' and symbol in watchlist:
                watchlist.remove(symbol)
                
            session['watchlist'] = watchlist
            return jsonify(success=True, watchlist=watchlist)
            
        except Exception as e:
            return jsonify(success=False, error=str(e))
    
    # GET request returns current watchlist
    return jsonify(session.get('watchlist', []))

@app.route('/predict/<ticker>')
@cache.cached(timeout=3600)
def get_prediction(ticker):
    try:
        prediction = predict_future_prices(ticker)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
