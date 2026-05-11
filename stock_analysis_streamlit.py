import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

#page config
st.set_page_config(
    page_title = "Stock Dashboard",
    layout = "wide"
)

st.markdown("## STOCK MARKET DASHBOARD ")
st.markdown("ANALYZE STOCKS WITH ML , INDICATORS AND SIGNALS")

#sidebar
st.sidebar.header("settings")


stocks = st.sidebar.multiselect(
    "Select Stocks",
    ["AAPL","MSFT","GOOGL","TSLA","AMZN"],
    default=["AAPL"]

)
import datetime
start_date = st.date_input("Start date",datetime.date(2020,1,1))
end_date = st.sidebar.date_input("End date",datetime.date.today())

#fetch data
data = {}
for stock in stocks:
    data[stock] =  yf.download(stock,start = start_date,end = end_date)

#display
for stock in stocks:
    st.subheader(f"{stock} analysis")

    df = data[stock].copy()

    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

#buy / sell signals
    df['Signal'] = 0
    df.loc[df['RSI']<30, 'RSI_signal'] = 1#(oversold)
    df.loc[df['RSI']>70, 'RSI_signal'] = -1#(overbrought)

  # -------------------------------
# MACHINE LEARNING PREDICTION (FIXED)
# -------------------------------

    from sklearn.linear_model import LinearRegression

# Create target
    df['Prediction'] = df['Close'].shift(-1)

# Select required columns
    ml_df = df[['Close', 'MA_20', 'EMA_20', 'Prediction']]

# DROP NaN TOGETHER (IMPORTANT FIX)
    ml_df = ml_df.dropna()

# Split features and target
    X = ml_df[['Close', 'MA_20', 'EMA_20']]
    y = ml_df['Prediction']

# Train model
    model = LinearRegression()
    model.fit(X, y)

# Predict next day price
    last_data = df[['Close', 'MA_20', 'EMA_20']].iloc[-1].values.reshape(1, -1)
    predicted_price = model.predict(last_data)[0]
#metrics

    col1, col2, col3 = st.columns(3)

    if not df.empty:
        latest_price = df['Close'][stock].iloc[-1]
        highest_price = df['Close'][stock].max()
        lowest_price = df['Close'][stock].min()

        col1.metric("Latest Price", f"${latest_price:.2f}")
        col2.metric("Highest", f"${highest_price:.2f}")
        col3.metric("Lowest", f"${lowest_price:.2f}")


        

#graph
        fig, ax = plt.subplots(figsize=(10,5))

        ax.plot(df['Close'],label="Close",linewidth=2)
        ax.plot(df['MA_20'],label="MA 20")
        ax.plot(df['EMA_20'],label = "EMA 20")
   
   #buy 
        ax.scatter(df.index[df["Signal"]==1],
              df['Close'][df['Signal']==1],
              marker="^",s=100,label="buy")
   #sell
        ax.scatter(df.index[df["Signal"]==1],
              df['Close'][df['Signal']==1],
              marker="v",s=100,label="sell")
        ax.legend()
        ax.set_title(stock)

        st.pyplot(fig)
    else:
        st.warning("no data available")

   #data atble
    with st.expander("view data"):
       st.dataframe(df.tail())
   