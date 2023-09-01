import yfinance as yf
import pandas as pd
import sqlite3

btc = yf.Ticker('BTC-USD')
prices = btc.history(period='5y')
prices.drop(columns=['Open', 'High', 'Low', 'Dividends', 'Stock Splits'], axis = 1, inplace=True)

print(prices.head())

conn = sqlite3.connect('Database.db')
prices.to_sql(name='PRICE', con=conn, index=True, if_exists='replace')