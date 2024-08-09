import tkinter as tk
from tkinter import ttk, messagebox
from src.data_collection import get_stock_data
from src.preprocessing import preprocess_data, add_features
from src.model import prepare_data, train_model, evaluate_model, plot_predictions
from src.model_lstm import prepare_data_lstm, build_lstm_model, evaluate_model_lstm, plot_predictions_lstm
from src.eda import plot_candlestick, plot_interactive
import os
import pandas as pd

def run_analysis():
    ticker = ticker_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    model_choice = model_var.get()

    if not ticker or not start_date or not end_date:
        messagebox.showerror("input error", "all fields are required!")
        return

    data = get_stock_data(ticker, start_date, end_date)
    if not os.path.exists('data'):
        os.makedirs('data')
    data.to_csv(f'data/{ticker}_stock_data.csv')

    data = preprocess_data(f'data/{ticker}_stock_data.csv')
    data = add_features(data)
    data.to_csv(f'data/{ticker}_stock_data_features.csv')

    if model_choice == 'RandomForest':
        X_train, X_test, y_train, y_test = prepare_data(f'data/{ticker}_stock_data_features.csv')
        model = train_model(X_train, y_train)
        predictions, rmse = evaluate_model(model, X_test, y_test)
        plot_predictions(data, predictions, ticker)
    elif model_choice == 'LSTM':
        X_train, X_test, y_train, y_test, scaler = prepare_data_lstm(f'data/{ticker}_stock_data_features.csv')
        model = build_lstm_model(X_train, y_train)
        predictions, rmse = evaluate_model_lstm(model, X_test, y_test, scaler)
        plot_predictions_lstm(data, predictions, ticker)

    result_label.config(text=f"RMSE: {rmse}")

def visualize_candlestick():
    ticker = ticker_entry.get()
    filepath = f'data/{ticker}_stock_data_features.csv'
    if not os.path.exists(filepath):
        messagebox.showerror("file error", f"no data found for {ticker}. please run analysis first.")
        return
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    plot_candlestick(data, ticker)

def visualize_interactive():
    ticker = ticker_entry.get()
    filepath = f'data/{ticker}_stock_data_features.csv'
    if not os.path.exists(filepath):
        messagebox.showerror("file error", f"no data found for {ticker}. please run analysis first.")
        return
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    plot_interactive(data, ticker)

root = tk.Tk()
root.title("Trade Tracker")

ttk.Label(root, text="Stock Ticker: ").grid(row=0, column=0, padx=10, pady=10)
ticker_entry = ttk.Entry(root)
ticker_entry.grid(row=0, column=1, padx=10, pady=10)

ttk.Label(root, text="Start Date (YYYY-MM-DD): ").grid(row=1, column=0, padx=10, pady=10)
start_date_entry = ttk.Entry(root)
start_date_entry.grid(row=1, column=1, padx=10, pady=10)

ttk.Label(root, text="End Date (YYYY-MM-DD): ").grid(row=2, column=0, padx=10, pady=10)
end_date_entry = ttk.Entry(root)
end_date_entry.grid(row=2, column=1, padx=10, pady=10)

ttk.Label(root, text="Choose Model: ").grid(row=3, column=0, padx=10, pady=10)
model_var = tk.StringVar(value='RandomForest')
model_rf = ttk.Radiobutton(root, text="RandomForest", variable=model_var, value='RandomForest')
model_rf.grid(row=3, column=1, padx=10, pady=5)
model_lstm = ttk.Radiobutton(root, text="LSTM", variable=model_var, value='LSTM')
model_lstm.grid(row=4, column=1, padx=10, pady=5)

analyze_button = ttk.Button(root, text="Run Analysis", command=run_analysis)
analyze_button.grid(row=5, column=0, columnspan=2, pady=10)

candlestick_button = ttk.Button(root, text="Candlestick Chart", command=visualize_candlestick)
candlestick_button.grid(row=6, column=0, columnspan=2, pady=10)

interactive_button = ttk.Button(root, text="Interactive Plot", command=visualize_interactive)
interactive_button.grid(row=7, column=0, columnspan=2, pady=10)

result_label = ttk.Label(root, text="")
result_label.grid(row=8, column=0, columnspan=2, pady=10)

root.mainloop()