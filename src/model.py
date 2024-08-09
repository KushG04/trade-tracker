import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def prepare_data(filepath):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    features = data[['Close', 'MA_20', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'BB_Mid', 'BB_Upper', 'BB_Lower']]
    target = data['Close'].shift(-1)
    X = features[:-1]
    y = target[:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_model(X_train, y_train):
    model = hyperparameter_tuning(X_train, y_train)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return predictions, rmse

def plot_predictions(data, predictions, ticker):
    plt.figure(figsize=(14,7))
    plt.plot(data.index[-len(predictions):], data['Close'][-len(predictions):], label='Actual Price')
    plt.plot(data.index[-len(predictions):], predictions, label='Predicted Price')
    plt.title(f'Actual vs Predicted {ticker} Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ticker = input("Stock Ticker: ")
    filepath = f'data/{ticker}_stock_data_features.csv'
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    X_train, X_test, y_train, y_test = prepare_data(filepath)
    model = train_model(X_train, y_train)
    predictions, rmse = evaluate_model(model, X_test, y_test)
    print(f'RMSE: {rmse}')
    plot_predictions(data, predictions, ticker)