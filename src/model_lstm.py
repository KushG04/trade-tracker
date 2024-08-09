import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input #type: ignore

def prepare_data_lstm(filepath, time_step=60):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    close_data = data['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test, scaler

def build_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    return model

def make_predictions(model, X_test, time_step):
    predictions = []
    current_input = X_test[0]

    for _ in range(len(X_test)):
        pred = model.predict(current_input.reshape(1, time_step, 1))
        predictions.append(pred[0])
        current_input = np.append(current_input[1:], pred)
    
    return np.array(predictions).reshape(-1, 1)

def evaluate_model_lstm(model, X_test, y_test, scaler):
    predictions = make_predictions(model, X_test, X_test.shape[1])
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    return predictions, rmse

def plot_predictions_lstm(data, predictions, ticker):
    plt.figure(figsize=(14,7))
    train_data = data[:int(len(data)*0.8)]
    valid_data = data[int(len(data)*0.8):].copy()

    if len(predictions) > len(valid_data):
        predictions = predictions[:len(valid_data)]
    elif len(predictions) < len(valid_data):
        valid_data = valid_data.iloc[:len(predictions)]

    valid_data['Predictions'] = predictions

    plt.plot(train_data['Close'], label='Train')
    plt.plot(valid_data[['Close', 'Predictions']], label=['Valid', 'Predictions'])
    plt.title(f'Actual vs Predicted {ticker} Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ticker = input("Stock Ticker: ")
    filepath = f'data/{ticker}_stock_data_features.csv'
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    X_train, X_test, y_train, y_test, scaler = prepare_data_lstm(filepath)
    model = build_lstm_model(X_train, y_train)
    predictions, rmse = evaluate_model_lstm(model, X_test, y_test, scaler)
    print(f'RMSE: {rmse}')
    plot_predictions_lstm(data, predictions, ticker)