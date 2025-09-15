# Stock Price Prediction with LSTM & Flask

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Framework](https.img.shields.io/badge/Framework-Flask-black.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%7C%20Keras-orange.svg)
![Libraries](https.img.shields.io/badge/Libraries-Pandas%20%7C%20yfinance%20%7C%20Matplotlib-green.svg)


This project is a comprehensive web application that demonstrates the use of a Long Short-Term Memory (LSTM) neural network to forecast stock prices. Built with Flask, the application provides an interactive interface for users to analyze historical stock data and view model-based predictions.

---

## ✨ Key Features

-   📈 **Dynamic Stock Analysis**: Fetches and processes the latest 10 years of stock data for any user-provided ticker using the `yfinance` library.
-   🧠 **Deep Learning Prediction**: Utilizes a sophisticated LSTM model built with Keras to forecast future stock prices based on historical patterns.
-   📊 **Rich Data Visualization**: Generates and displays multiple plots using Matplotlib, including:
    -   Historical Closing Prices.
    -   100-Day & 200-Day Simple Moving Averages (SMA).
    -   A comparison chart of Original vs. Predicted prices.
-   🌐 **Interactive Web Interface**: A clean and user-friendly UI built with Flask allows for easy interaction and clear presentation of results.

---

## 🛠️ How It Works

The project follows a standard machine learning workflow, separating model training from application deployment.

### 1. Model Training (`LSTM_Model.ipynb`)

The LSTM model is trained offline in a Jupyter Notebook.
1.  **Data Collection**: Downloads 10 years of historical stock data for a target company (e.g., Apple, `AAPL`).
2.  **Preprocessing**: The closing prices are scaled between 0 and 1 to optimize model training. The data is then structured into sequences, where each input is a 100-day window of prices used to predict the price on the 101st day.
3.  **Model Architecture**: A sequential Keras model is constructed with multiple LSTM layers and Dropout layers to prevent overfitting.
4.  **Training & Export**: The model is trained on the prepared data, and the final trained weights are saved to `keras_model.h5`.

### 2. Web Application (`app.py`)

The Flask application serves the model and user interface.
1.  **Model Loading**: The pre-trained `keras_model.h5` is loaded once at startup.
2.  **User Input**: A user submits a stock ticker via the web form.
3.  **Real-time Processing**: The app fetches live data for the ticker, calculates moving averages, and prepares the data for prediction using the same scaling and sequencing logic from training.
4.  **Forecasting**: The processed data is fed into the loaded LSTM model to generate a price forecast.
5.  **Rendering**: All data and generated charts are dynamically rendered on the web page.

---

## 💡 Key Concepts Demonstrated

This project showcases several important concepts in data science and deep learning:

-   **Time-Series Forecasting**: Using historical data points to predict future values.
-   **Recurrent Neural Networks (RNNs)**: Specifically **LSTMs**, which are designed to recognize patterns in sequential data, making them ideal for time-series tasks.
-   **Data Normalization**: Scaling data is a critical preprocessing step for neural networks to ensure stable and efficient training.
-   **Feature Engineering**: Structuring raw data into input/output sequences (100-day windows) for supervised learning.
-   **Model Deployment**: Serving a trained deep learning model through a web framework (Flask).

---

## ⚠️ Important Disclaimer

> The included LSTM model (`keras_model.h5`) was trained **exclusively on Apple Inc. (AAPL) stock data**. While the application allows you to input any ticker, the predictions for other stocks are for **demonstration purposes only** and should not be used for financial decisions. A model's predictions are only reliable for the specific data distribution it was trained on.

---

## 🚀 Setup and Usage

### Prerequisites
-   Python 3.8+
-   `pip` and `venv` (or a similar package manager like `uv`)

### 1. Clone the Repository & Navigate
```bash
git clone [https://github.com/your-username/stock-prediction-lstm.git](https://github.com/your-username/stock-prediction-lstm.git)
cd stock-prediction-lstm
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### 3. Run the Application
Ensure the `keras_model.h5` file (generated from the training notebook) is in the same directory as `app.py`.

```bash
cd FlaskApp/
#and then run either of bellow command
flask run
# Or
python app.py
```
The application will be available at `http://127.0.0.1:5000`.

---

## 📂 Project Structure

```
.
├── requirements.txt                            # Python dependencies
├── Streamlit                                   # StreamLit App 
├── FlaskApp                                    # Flask App with Temlate Rendering
    └── app.py                                  # Main Flask application script
    ├── keras_model.h5                          # Pre-trained LSTM model
    ├── LSTM_Model.ipynb                        # Notebook for model training
    └── templates/
        └── home.html                           # HTML template for the UI
```