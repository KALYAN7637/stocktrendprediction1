import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
import streamlit as st
import datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Title for the app
st.title('STOCK TREND PREDICTION USING LSTM ON GLOBAL STOCK MARKETS')

# User input for stock ticker and date range
ticker = st.text_input('Enter Ticker Name', 'KOTAKBANK.NS')
start = st.date_input('Start', value=pd.to_datetime('2010-01-01'))
end = st.date_input('End', value=pd.to_datetime('today'))

# Download stock data
df = yf.download(ticker, start, end)

# Show dataframe
st.subheader('Data From Start Date To Today')
st.dataframe(df, width=700)

# Display stock charts
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

# Moving Averages
st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma200)
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df['Close'])
st.pyplot(fig)

# Splitting data into training and testing
data_training = df['Close'][:int(len(df) * 0.8)]
data_testing = df['Close'][int(len(df) * 0.2):]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_scaled = scaler.fit_transform(data_training.values.reshape(-1, 1))

# Prepare training data
x_train = []
y_train = []
for i in range(100, data_training_scaled.shape[0]):
    x_train.append(data_training_scaled[i-100:i])
    y_train.append(data_training_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load the pre-trained model
model = load_model('keras_model.h5')

# Prepare testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing])
input_data = scaler.transform(final_df.values.reshape(-1, 1))

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot the predictions vs original price
st.subheader('Prediction vs Original Price')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.plot(y_test, 'b', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Price Movements and Annual Return
df2 = df.copy()
df2['% Change'] = df['Close'] / df['Close'].shift(1) - 1
df2.dropna(inplace=True)
st.write(df)
annual_return = df2['% Change'].mean() * 252 * 100
st.write('Annual Return is', annual_return, '%')

# Predictions for future days
n = st.number_input("Enter the number of days to predict", min_value=1, step=1)
x_extended = np.arange(len(y_test) + n)

# Plot future predictions
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.plot(y_test, 'b', label='Original Price')
plt.plot(x_extended[-n:], y_predicted[-n:], 'g', label=f'Next {n} Days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Make predictions for the next n days
x_pred = x_test[-1]
predicted_prices = []
for _ in range(n):
    pred = model.predict(np.array([x_pred]))
    predicted_prices.append(pred[0])
    x_pred = np.roll(x_pred, -1, axis=0)
    x_pred[-1] = pred[0]

predicted_prices = np.array(predicted_prices)
predicted_prices = predicted_prices.reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_prices)  # Inverse transform to original scale

# Add predicted prices to the DataFrame
predicted_dates = pd.date_range(start=df.index[-1] + dt.timedelta(days=1), periods=n, freq='B')
predicted_df = pd.DataFrame(predicted_prices, index=predicted_dates, columns=['Predicted Price'])

# Plot using Plotly
fig = px.line(df, x=df.index, y='Close', title=ticker)
fig.add_scatter(x=predicted_dates, y=predicted_prices.flatten(), mode='lines', name='Predicted Price')
st.plotly_chart(fig)

# ADO Indicator
def calculate_ado(df):
    mf_volume = df['Volume'] * ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])).fillna(0)
    ado = np.cumsum(mf_volume)
    return ado

def plot_ado_close(df, ado, start_date, end_date):
    df = df.loc[start_date:end_date]
    df['ADO'] = ado
    fig = px.line(df, x=df.index, y=['Close', 'ADO'], labels={'value': 'Price', 'variable': 'Indicator', 'index': 'Date'}, title='Accumulation And Distribution Oscillator (ADO) Indicator')
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

ado = calculate_ado(df)
plot_ado_close(df, ado, start, end)

# MACD Indicator
def plot_macd_graph(df):
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    colors = ['red' if val < 0 else 'green' for val in df['Histogram']]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal', line=dict(color='pink')))
    fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker=dict(color=colors)))
    fig.update_layout(xaxis_rangeslider_visible=True, height=500)
    st.plotly_chart(fig)

plot_macd_graph(df)

# WMA vs Closing Price
def weighted_moving_average(prices, weights):
    return sum(prices * weights) / sum(weights)

wma_period_13 = 13
weights_13 = list(range(1, wma_period_13 + 1))
df['WMA_13'] = df['Close'].rolling(window=wma_period_13).apply(lambda prices: weighted_moving_average(prices, weights_13))

fig = px.line(df, x=df.index, y=['Close', 'WMA_13'], title='WMA and Closing Price')
fig.update_traces(line=dict(color='blue'), selector=dict(name='Close'))
fig.update_traces(line=dict(color='red'), selector=dict(name='WMA_13'))
fig.update_layout(xaxis_rangeslider_visible=True, height=500)
st.plotly_chart(fig)

# EMA vs Closing Price
def exponential_moving_average(prices, period):
    alpha = 2 / (period + 1)
    ema = [prices[0]]
    for i in range(1, len(prices)):
        ema.append((prices[i] - ema[-1]) * alpha + ema[-1])
    return ema

ema_period_50 = 50
ema_period_100 = 100
ema_period_200 = 200

df['EMA_50'] = exponential_moving_average(df['Close'], ema_period_50)
df['EMA_100'] = exponential_moving_average(df['Close'], ema_period_100)
df['EMA_200'] = exponential_moving_average(df['Close'], ema_period_200)

fig = px.line(df, x=df.index, y=['Close', 'EMA_50', 'EMA_100', 'EMA_200'], title='Exponential Moving Averages (EMA) and Closing Price')
fig.update_traces(line=dict(color='blue'), selector=dict(name='Close'))
fig.update_traces(line=dict(color='red'), selector=dict(name='EMA_50'))
fig.update_traces(line=dict(color='green'), selector=dict(name='EMA_100'))
fig.update_traces(line=dict(color='orange'), selector=dict(name='EMA_200'))
fig.update_layout(xaxis_rangeslider_visible=True, height=500)
st.plotly_chart(fig)
