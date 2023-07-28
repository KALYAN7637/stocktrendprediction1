import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader  as data
from keras.models import load_model
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import plotly_express as px
import sklearn as sk
#import talib as ta
import streamlit as st
import  datetime as dt
#from talib import SMA,EMA
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
#import talib as ta
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import  datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from plotly import graph_objs as go



st.title('STOCK TREND PREDICTION USING LSTM ON GLOBAL STOCK MARKETS')
ticker=st.text_input('Enter Ticker Name','KOTAKBANK.NS')

start=st.date_input('Start', value=pd.to_datetime('2010-01-01'))
end=st.date_input('End', value=pd.to_datetime('today'))


df=yf.download(ticker,start,end)
fig= px.line(df,x=df.index, y=df['Close'], title=ticker)
st.plotly_chart(fig)

st.subheader('Data From Start Date To Today')
st.dataframe(df,width=700)

st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Open)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


# Splitting data into training and testing
data_training = df['Close'][:int(len(df) * 0.8)]
data_testing = df['Close'][int(len(df) * 0.2):]


# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_scaled = scaler.fit_transform(data_training.values.reshape(-1, 1))


#print(data_training.shape)
#print(data_testing.shape)



#from sklearn.preprocessing import MinMaxScaler
#scaler=MinMaxScaler(feature_range=(0,1))


# Prepare training data
x_train = []
y_train = []
for i in range(100, data_training_scaled.shape[0]):
    x_train.append(data_training_scaled[i-100:i])
    y_train.append(data_training_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load the trained model
model = load_model(r'C:\Users\KALYAN\Desktop\all projects\experiments\keras_model.h5')



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

# Plot the predictions
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.plot(y_test, 'b', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# Get user input for the number of days to predict
n = st.number_input("Enter the number of days to predict", min_value=1, step=1)

# Extend the x-axis for the prediction period
x_extended = np.arange(len(y_test) + n)

# Plot the predictions
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.plot(y_test, 'b', label='Original Price')
plt.plot(x_extended[-n:], y_predicted[-n:], 'g', label=f'Next {n} Days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)





# Extend the x-axis for the prediction period
x_extended = np.arange(len(df))

# Make predictions for the next n days
x_pred = x_test[-1]
predicted_prices = [] 
for _ in range(n):
    pred = model.predict(np.array([x_pred]))
    predicted_prices.append(pred[0])
    #x_pred = np.concatenate((x_pred[1:], np.expand_dims(pred[0], axis=0)))
    x_pred = np.roll(x_pred, -1, axis=0)
    x_pred[-1] = pred[0]

predicted_prices = np.array(predicted_prices)
predicted_prices = predicted_prices.reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_prices)  # Scale back to original prices

# Add predicted prices to the DataFrame
predicted_dates = pd.date_range(start=df.index[-1] + dt.timedelta(days=1), periods=n, freq='B')
predicted_df = pd.DataFrame(predicted_prices, index=predicted_dates, columns=['Predicted Price'])

# Plot the graph using Plotly Express
fig = px.line(df, x=df.index, y='Close', title=ticker)
fig.add_scatter(x=predicted_dates, y=predicted_prices.flatten(), mode='lines', name='Predicted Price')
st.plotly_chart(fig)









st.subheader('Accumulation And Distribution Oscillator Indicator')

def calculate_ado(df):
    mf_volume = df['Volume'] * ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])).fillna(0)
    ado = np.cumsum(mf_volume)
    return ado

def plot_ado_close(df, ado, start_date, end_date):
    df = df.loc[start_date:end_date]

    fig_ado = px.line(df, x=df.index, y=ado)
    fig_ado.update_layout(
        xaxis_title='Date',
        yaxis_title='ADO'
    )
    fig_ado.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_ado)

    fig_close = px.line(df, x=df.index, y='Close')
    fig_close.update_layout(
        title='Closing Price',
        xaxis_title='Date',
        yaxis_title='Price'
    )
    #fig_close.update_xaxes(rangeslider_visible=True)
    #st.plotly_chart(fig_close)

def main():
   
    ado = calculate_ado(df)
    plot_ado_close(df, ado, start,end)

if __name__ == '__main__':
    main()




def calculate_ado(df):
    mf_volume = df['Volume'] * ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])).fillna(0)
    ado = np.cumsum(mf_volume)
    return ado

def plot_ado_close(df, ado, start_date, end_date):
    df = df.loc[start_date:end_date]
    df['ADO'] = ado

    fig = px.line(df, x=df.index, y=['Close', 'ADO'])
    fig.update_layout(
        title='Accumulation And Distribution Oscillator (ADO) Indicator',
        xaxis_title='Date',
        yaxis_title='Value'
    )
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="ADO", secondary_y=True)
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)









st.subheader("MACD Indicator")
df = yf.download(ticker, start, end)
def plot_macd_graph(df):
    # Calculate MACD
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Histogram'] = df['MACD'] - df['Signal']

    # Determine the color for the histogram bars
    colors = ['red' if val < 0 else 'green' for val in df['Histogram']]

    # Create a single plot
    fig = go.Figure()

    # Add MACD line
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD',line=dict(color='cyan')))

    # Add MACD signal line
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal',line=dict(color='pink')))

    # Add MACD histogram with custom colors
    fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker=dict(color=colors)))

    # Update layout
    fig.update_layout(xaxis_rangeslider_visible=True,height=500)

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)


df = yf.download(ticker, start, end)

if df.empty:
    st.warning('No data available for the selected symbol and date range')
else:
    plot_macd_graph(df)





st.subheader('WMA VS CLOSING PRICE')
wma_period_13 = 13 
close_prices = df['Close'].values
wma_13 = ta.WMA(close_prices, timeperiod=wma_period_13)
df['WMA_13'] = wma_13
fig = px.line(df, x=df.index, y=['Close', 'WMA_13'], title='WMA and Closing Price')  # Include 'WMA_13' in y
fig.update_traces(line=dict(color='red'), selector=dict(name='WMA_13'))  # Use 'WMA_13' as the selector
fig.update_layout(xaxis_title='Date', yaxis_title='Price', title_text='Time Series Data',
                  xaxis_rangeslider_visible=True,height=500)
st.plotly_chart(fig)



st.subheader('EMA VS CLOSING PRICE')
ema_period_50 = 50  
ema_period_100 = 100
ema_period_200 = 200
close_prices = df['Close'].values
ema_50 = ta.EMA(close_prices, timeperiod=ema_period_50)
ema_100 = ta.EMA(close_prices, timeperiod=ema_period_100)
ema_200 = ta.EMA(close_prices, timeperiod=ema_period_200)
df['EMA_50'] = ema_50  
df['EMA_100'] = ema_100
df['EMA_200'] = ema_200
fig = px.line(df, x=df.index, y=['Close', 'EMA_50', 'EMA_100','EMA_200'], title='EMA and Closing Price')
fig.update_traces(line=dict(color='red'), selector=dict(name='EMA_50'))
fig.update_traces(line=dict(color='pink'), selector=dict(name='EMA_100'))
fig.update_traces(line=dict(color='green'), selector=dict(name='EMA_200'))
fig.update_layout(xaxis_title='Date', yaxis_title='Price')
fig.layout.update(title_text='Time Series Data',xaxis_rangeslider_visible=True,height=500)
st.plotly_chart(fig)





st.subheader('50EMA VS CLOSING PRICE')
ema_period_50 = 50  
close_prices = df['Close'].values
ema_50 = ta.EMA(close_prices, timeperiod=ema_period_50)
df['EMA_50'] = ema_50  
fig = px.line(df, x=df.index, y=['Close', 'EMA_50'], title='EMA50 and Closing Price')
fig.update_traces(line=dict(color='red'), selector=dict(name='EMA_50'))
fig.update_layout(xaxis_title='Date', yaxis_title='Price')
fig.layout.update(title_text='Time Series Data',xaxis_rangeslider_visible=True,height=500)
st.plotly_chart(fig)




st.subheader('Bollinger Bands Indicator vs Closing price')

def plot_macd_bollinger_candlestick(df):
    # Calculate MACD
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Histogram'] = df['MACD'] - df['Signal']

    # Calculate Bollinger Bands
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['std'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['SMA'] + 2 * df['std']
    df['Lower'] = df['SMA'] - 2 * df['std']

    # Create a single plot
    fig = go.Figure()

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Upper Band', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Lower Band', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], mode='lines', name='SMA', line=dict(color='blue')))

    # Add MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal', line=dict(color='blue')))
    fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker=dict(color='green')))

    # Add candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick'))

    # Update layout
    fig.update_layout(xaxis_rangeslider_visible=True, height=500)  # Increase the height of the graph by three times

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)



df = yf.download(ticker, start, end)

if df.empty:
    st.warning('No data available for the selected symbol and date range.')
else:
    plot_macd_bollinger_candlestick(df)





st.subheader('bollinger bands vs closing price')    
def plot_macd_bollinger_candlestick(df):
    # Calculate MACD
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Histogram'] = df['MACD'] - df['Signal']

    # Calculate Bollinger Bands
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['std'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['SMA'] + 2 * df['std']
    df['Lower'] = df['SMA'] - 2 * df['std']

    # Create a single plot
    fig = go.Figure()

    # Add Bollinger Bands (Upper and Lower)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Upper Band', line=dict(color='pink')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Lower Band', line=dict(color='red')))

    # Add Bollinger Bands (Fill between Upper and Lower)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], fill=None, mode='lines', line=dict(color='gray'), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], fill='tonexty', mode='lines', line=dict(color='gray'), showlegend=False))

    # Add Bollinger Bands (Middle line - SMA)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], mode='lines', name='SMA', line=dict(color='blue')))

    # Add MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal', line=dict(color='blue')))
    fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker=dict(color='green')))

    # Add candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick'))
    fig.update_layout(xaxis_rangeslider_visible=True, height=500)

    # Update layout
    fig.update_layout(xaxis_rangeslider_visible=True)

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)


if df.empty:
    st.warning('No data available for the selected symbol and date range.')
else:
    plot_macd_bollinger_candlestick(df)















def plot_bollinger_bands(df):
    # Calculate Bollinger Bands
    df['MA'] = df['Close'].rolling(window=20).mean()
    df['std'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA'] + 2 * df['std']
    df['Lower'] = df['MA'] - 2 * df['std']

    # Create a single plot
    fig = go.Figure()

    # Add Bollinger Bands
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick'), row=1, col=1)

    
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'), row=2, col=1)   
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal'), row=2, col=1)  
    fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram'), row=2, col=1)





    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Upper Band', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Lower Band', line=dict(color='red')))

    # Add closing price
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='blue')))

    # Update layout
    fig.update_layout(xaxis_rangeslider_visible=True)

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)


#if df.empty:
    st.warning('No data available for the selected symbol and date range.')
#else:
    plot_bollinger_bands(df)








st.subheader('Price Movements')
df2=df
df2['% Change']=df['Close']/df['Close'].shift(1) - 1
df2.dropna(inplace = True)
st.write(df)
annual_return=df2['% Change'].mean()*252*100
st.write('Annual Return is', annual_return,'%')


