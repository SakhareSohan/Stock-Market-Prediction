from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import yfinance as yf
import base64
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# --- Load the Model ---
# Use a relative path so the app is portable.
# The model should be in the same directory as this script.
MODEL_PATH = 'keras_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure 'keras_model.h5' is in the root directory.")
model = load_model(MODEL_PATH)

def get_base64_chart_image(fig):
    """Converts a matplotlib figure to a base64 encoded PNG image."""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) # Close the figure to free up memory
    return chart_image

def process_stock_data(ticker_symbol):
    """
    Fetches stock data, generates charts, and makes predictions.
    This function consolidates all the logic to avoid code duplication.
    """
    stock_data = yf.Ticker(ticker_symbol).history(period="10y")
    if stock_data.empty:
        return None # Return None if ticker is invalid or no data is found

    # --- Basic Data and Stats ---
    table_html = stock_data.head().to_html(classes='table table-striped text-center', justify='center')
    describe_html = stock_data.describe().to_html(classes='table table-striped text-center', justify='center')

    # --- Chart Generation ---
    charts = {}
    
    # Chart 1: Closing Price
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'])
    plt.title(f'{ticker_symbol} Closing Price History')
    plt.xlabel('Date'); plt.ylabel('Price (USD)')
    charts['chart_image1'] = get_base64_chart_image(fig1)

    # Chart 2 & 3: Moving Averages
    ma100 = stock_data['Close'].rolling(100).mean()
    ma200 = stock_data['Close'].rolling(200).mean()
    
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.plot(ma100, 'r', label='100-Day MA')
    plt.title(f'{ticker_symbol} Price vs 100-Day Moving Average')
    plt.xlabel('Date'); plt.ylabel('Price (USD)'); plt.legend()
    charts['chart_image2'] = get_base64_chart_image(fig2)

    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.plot(ma100, 'r', label='100-Day MA')
    plt.plot(ma200, 'g', label='200-Day MA')
    plt.title(f'{ticker_symbol} Price vs Moving Averages')
    plt.xlabel('Date'); plt.ylabel('Price (USD)'); plt.legend()
    charts['chart_image3'] = get_base64_chart_image(fig3)

    # --- Prediction Logic ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_testing = pd.DataFrame(stock_data['Close'])
    
    # Scale all data for consistent processing
    scaled_data = scaler.fit_transform(data_testing)

    x_test, y_test = [], []
    for i in range(100, len(scaled_data)):
        x_test.append(scaled_data[i-100:i, 0])
        y_test.append(scaled_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_predicted = model.predict(x_test)

    # Inverse transform to get actual price values
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor
    
    latest_predicted_price = y_predicted[-1][0]

    # Chart 4: Prediction vs Original
    fig4 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.title(f'{ticker_symbol} Price Prediction')
    plt.xlabel('Time'); plt.ylabel('Price (USD)'); plt.legend()
    charts['chart_image4'] = get_base64_chart_image(fig4)

    return {
        'user_input': ticker_symbol,
        'table': table_html,
        'describe': describe_html,
        'charts': charts,
        'latest_predicted_price': f"{latest_predicted_price:.2f}"
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.get('ticker', 'AAPL').upper()
    else:
        user_input = 'AAPL' # Default on GET request
    
    data = process_stock_data(user_input)
    
    if data is None:
        return render_template("error.html", ticker=user_input)

    return render_template("home.html", **data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)