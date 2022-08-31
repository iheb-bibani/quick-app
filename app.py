import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import pandas_datareader as pdr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import talib as ta


st.markdown("<h1 style='text-align: center; color: red;'>Quick Switzerland APP</h1>", unsafe_allow_html=True)

tickers = ['AAPL', 'MSFT', 'FB']

st.sidebar.header('Parameters')

x = st.sidebar.selectbox('Tickers', ('AAPL', 'MSFT', 'FB'))

if x == 'AAPL':

    df = pdr.DataReader(tickers[0], 'yahoo')

elif x == 'MSFT':

    df = pdr.DataReader(tickers[1], 'yahoo')

else:

    df = pdr.DataReader(tickers[2], 'yahoo')

def input_parameters():

    sma_1 = st.sidebar.slider('SMA_1', 0, 30, 15)
    sma_2 = st.sidebar.slider('SMA_2', 0, 30, 15)
    rsi = st.sidebar.slider('RSI', 0, 30, 15)
    atr = st.sidebar.slider('ATR', 0, 30, 15)

    data = {
        'sma_10': ta.SMA(df['Close'], sma_1),
        'sma_20': ta.SMA(df['Close'], sma_2),
        'rsi': ta.RSI(df['Close'], rsi),
        'atr': ta.ATR(df['High'], df['Low'], df['Close'], atr)

    }

    features = pd.DataFrame(data)

    return features

df.index = df.index.strftime("%Y/%m/%d")

input = pd.DataFrame(input_parameters())

st.subheader('**Data Plot**')

st.line_chart(df['Close'])

st.subheader('**Data Exploration**')

st.markdown('**First 5 rows**')
st.table(df.head())

st.markdown('**Last 5 rows**')
st.table(df.tail())

st.markdown('**Is there any missing values ?**')
st.text(df.isna().sum())

st.markdown('**Some Descriptive statistics**')
st.table(df.describe())

st.subheader('Predictions')

X = input[30:]
y = df['Close'][30:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
predictions = rf.predict(X)

st.line_chart(predictions)
