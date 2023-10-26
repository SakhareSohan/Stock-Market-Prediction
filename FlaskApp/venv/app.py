from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import yfinance as yf
import base64
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
import json

app = Flask(__name__)

user_input = 'AAPL'
model = load_model('D:/FlaskApp/venv/keras_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.get('ticker')
        aapl = yf.Ticker(user_input)
        df = aapl.history(period="10y")
        table_html = df.head().to_html(border=None)

        describe = df.describe()

        ma100 = df['Close'].rolling(window=100).mean()
        ma200 = df['Close'].rolling(window=200).mean()

        # Create chart 1: Closing Price vs Time Chart
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'])
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Closing Price vs Time Chart')
        chart_image1 = get_base64_chart_image(fig1)

        # Create chart 2: Closing Price vs Time Chart with 100MA
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100-day MA')
        plt.plot(df['Close'], label='Closing Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Closing Price vs Time Chart with 100MA')
        plt.legend()
        chart_image2 = get_base64_chart_image(fig2)

        # Create chart 3: Closing Price vs Time Chart with 100MA & 200MA
        fig3 = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100-day MA')
        plt.plot(ma200, label='200-day MA')
        plt.plot(df['Close'], label='Closing Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Closing Price vs Time Chart with 100MA & 200MA')
        plt.legend()
        chart_image3 = get_base64_chart_image(fig3)

        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
        scaler = MinMaxScaler(feature_range=(0,1))
        

        # Prepare input data for prediction
        past_100_days = data_training.tail(100)
        final_df = past_100_days.append(data_testing, ignore_index = True)
        input_data = scaler.fit_transform(final_df)
        x_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

    # Make predictions
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_
        scale_factor = 1/scaler[0]
        y_predicted = y_predicted *scale_factor
        y_test = y_test *scale_factor
        latest_predicted_price = y_predicted[-1]

    # Plot the chart
        fig4 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        chart_image4 = get_base64_chart_image(fig4)

        return render_template("home.html", describe = describe, user_input = user_input, table = table_html, chart_image1=chart_image1, chart_image2=chart_image2, chart_image3=chart_image3, chart_image4=chart_image4, latest_predicted_price=latest_predicted_price)

    else:
        aapl = yf.Ticker('AAPL')
        df = aapl.history(period="10y")
        table_html = df.head().to_html(border=None)

        describe = df.describe()

        ma100 = df['Close'].rolling(window=100).mean()
        ma200 = df['Close'].rolling(window=200).mean()

        # Create chart 1: Closing Price vs Time Chart
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'])
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Closing Price vs Time Chart')
        chart_image1 = get_base64_chart_image(fig1)

        # Create chart 2: Closing Price vs Time Chart with 100MA
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100-day MA')
        plt.plot(df['Close'], label='Closing Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Closing Price vs Time Chart with 100MA')
        plt.legend()
        chart_image2 = get_base64_chart_image(fig2)

        # Create chart 3: Closing Price vs Time Chart with 100MA & 200MA
        fig3 = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100-day MA')
        plt.plot(ma200, label='200-day MA')
        plt.plot(df['Close'], label='Closing Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Closing Price vs Time Chart with 100MA & 200MA')
        plt.legend()
        chart_image3 = get_base64_chart_image(fig3)

        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
        scaler = MinMaxScaler(feature_range=(0,1))
        data_training_array = scaler.fit_transform(data_training)

        # Prepare input data for prediction
        past_100_days = data_training.tail(100)
        final_df = past_100_days.append(data_testing, ignore_index = True)
        input_data = scaler.fit_transform(final_df)
        x_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_
        scale_factor = 1/scaler[0]
        y_predicted = y_predicted *scale_factor
        y_test = y_test *scale_factor
        latest_predicted_price = y_predicted[-1]

        # Plot the chart
        fig4 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        chart_image4 = get_base64_chart_image(fig4)

        return render_template("home.html", describe = describe, user_input = 'AAPL', table = table_html, chart_image1=chart_image1, chart_image2=chart_image2, chart_image3=chart_image3, chart_image4=chart_image4, latest_predicted_price=latest_predicted_price)
    
def get_base64_chart_image(fig):
    # Convert chart to base64-encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return chart_image

def get_news(ticker):
    ticker_info = yf.Ticker(ticker).info
    news = ticker_info['longBusinessSummary']
    return news

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)
