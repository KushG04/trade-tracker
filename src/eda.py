import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def plot_candlestick(data, ticker):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
    fig.update_layout(title=f'{ticker} Candlestick Chart',
                      xaxis_title='Date',
                      yaxis_title='Price')
    fig.show()

def plot_interactive(data, ticker):
    fig = px.line(data, x=data.index, y='Close', title=f'{ticker} Interactive Close Price')
    fig.add_scatter(x=data.index, y=data['MA_20'], mode='lines', name='20-Day MA')
    fig.add_scatter(x=data.index, y=data['Volatility'], mode='lines', name='Volatility')
    fig.show()

if __name__ == "__main__":
    ticker = input("Stock Ticker: ")
    filepath = f'data/{ticker}_stock_data_features.csv'
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    plot_candlestick(data, ticker)
    plot_interactive(data, ticker)