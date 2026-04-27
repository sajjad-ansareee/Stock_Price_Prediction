import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

# using analysis file functions
import data_analysis as da

# data loading with caching for performance
@st.cache_data
def load_data():
    df = pd.read_csv("./cleaned.csv")
    return df

# model loading with caching to avoid reloading on every interaction
@st.cache_resource
def load_model():
    model = joblib.load("./linear_regression_model.pkl")
    return model


df = load_data()
model = load_model()


# sidebar navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Market Analysis", "Price Prediction"]
)

# market analysis page
if page == "Market Analysis":

    st.title("Stock Market Analysis")

    st.write("Select a visualization to explore the market.")

    col1, col2, col3 = st.columns(3)

    # Card 1
    with col1:
        st.subheader("Market Trend")
        st.write("Visualize overall stock market movement.")
        trend_btn = st.button("View", key="trend")

    # Card 2
    with col2:
        st.subheader("Volume Analysis")
        st.write("Understand how trading volume behaves.")
        volume_btn = st.button("View", key="volume")

    # Card 3
    with col3:
        st.subheader("Price Distribution")
        st.write("Observe distribution of stock prices.")
        dist_btn = st.button("View", key="dist")

    st.divider()

    # plotting the graphs based on button clicks
    if trend_btn:
        st.header("Market Trend")

        fig = da.plot_market_trend(df)

        if fig is None:
            fig = plt.gcf()

        st.pyplot(fig)

        st.write(
            "This graph shows how the market price changes over time. "
            "It helps identify whether the market is generally increasing "
            "or decreasing."
        )

    if volume_btn:
        st.header("Volume Analysis")

        fig = da.plot_top_volume_stocks(df)

        if fig is None:
            fig = plt.gcf()

        st.pyplot(fig)

        st.write(
            "Trading volume represents the number of shares traded. "
            "Higher volume usually indicates stronger market interest "
            "in a particular stock."
        )

    if dist_btn:
        st.header("Price Distribution")

        fig = da.plot_returns_distribution(df)

        if fig is None:
            fig = plt.gcf()

        st.pyplot(fig)

        st.write(
            "Price distribution helps understand how stock prices are "
            "spread across the dataset and whether most prices are "
            "clustered within a specific range."
        )


# price prediction page
elif page == "Price Prediction":

    st.title("Stock Price Prediction")

    st.write("Enter today's stock values to predict the closing price.")

    # Symbol selection
    symbols = df["Symbol"].unique()
    symbol = st.selectbox("Select Stock Symbol", symbols)

    st.subheader("Input Features")

    open_price = st.number_input("Open", value=0.0)
    high_price = st.number_input("High", value=0.0)
    low_price = st.number_input("Low", value=0.0)
    volume = st.number_input("Volume", value=0.0)

    predict_btn = st.button("Predict")

    if predict_btn:

        features = np.array([[open_price, high_price, low_price, volume]])

        prediction = model.predict(features)

        predicted_price = float(prediction[0])

        st.success(
            f"The predicted closing price for tomorrow is: {predicted_price}"
        )