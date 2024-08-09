import pandas as pd

def preprocess_data(filepath):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    data.dropna(inplace=True)
    return data

def add_features(data):
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['Volatility'] = data['Close'].rolling(window=20).std()
    data = calculate_rsi(data)
    data = calculate_macd(data)
    data = calculate_bollinger_bands(data)
    return data

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

def calculate_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    data['MACD'] = macd
    data['MACD_Signal'] = signal
    return data

def calculate_bollinger_bands(data, window=20):
    data['BB_Mid'] = data['Close'].rolling(window=window).mean()
    data['BB_Upper'] = data['BB_Mid'] + 2 * data['Close'].rolling(window=window).std()
    data['BB_Lower'] = data['BB_Mid'] - 2 * data['Close'].rolling(window=window).std()
    return data

if __name__ == "__main__":
    ticker = input("Stock Ticker: ")
    filepath = f'data/{ticker}_stock_data.csv'
    data = preprocess_data(filepath)
    data = add_features(data)
    data.to_csv(f'data/{ticker}_stock_data_features.csv')