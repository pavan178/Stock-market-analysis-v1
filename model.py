import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

def create_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(ticker):
    # Get historical data
    stock = yf.Ticker(ticker)
    df = stock.history(period="3y")
    
    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
    
    # Create sequences
    x_train = []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Train model
    model = create_model()
    model.fit(x_train, scaled_data[60:], epochs=1, batch_size=32)
    
    # Predict next 7 days
    last_60_days = scaled_data[-60:]
    predictions = []
    for _ in range(7):
        x = last_60_days.reshape(1, 60, 1)
        pred = model.predict(x)
        predictions.append(pred[0][0])
        last_60_days = np.append(last_60_days[1:], pred)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    
    # Generate dates
    dates = [datetime.now() + timedelta(days=i) for i in range(1,8)]
    
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'prices': [float(p[0]) for p in predictions],
        'recommendation': 'Buy' if float(predictions[-1][0]) > float(df['Close'].iloc[-1]) else 'Sell',
        'confidence': float(np.abs(predictions[-1][0] - df['Close'].iloc[-1]) / df['Close'].iloc[-1])
    } 