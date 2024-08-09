import yfinance as yf

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

if __name__ == "__main__":
    ticker = input("Stock Ticker: ")
    start_date = input("Start Date (YYYY-MM-DD): ")
    end_date = input("End Date (YYYY-MM-DD): ")

    data = get_stock_data(ticker, start_date, end_date)
    data.to_csv(f'data/{ticker}_stock_data.csv')