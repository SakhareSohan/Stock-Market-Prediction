# Stock Price Prediction with LSTM & Flask

[Image of a stock market chart]

This project is a web application that predicts stock prices using a Long Short-Term Memory (LSTM) neural network. The application is built with Flask and allows users to input a stock ticker to view historical data, moving averages, and a price prediction chart.

## Features

-   **Interactive Web Interface**: A user-friendly UI built with Flask to input stock tickers.
-   **Deep Learning Model**: Utilizes an LSTM model trained on historical stock data to forecast future prices.
-   **Rich Data Visualization**: Generates and displays multiple charts using Matplotlib, including:
    -   Closing Price History
    -   100-Day and 200-Day Moving Averages
    -   Original vs. Predicted Price Comparison
-   **On-Demand Analysis**: Fetches the latest 10 years of stock data for any ticker using the `yfinance` library.

---

## How It Works

The project consists of two main parts:

### 1. Model Training (Jupyter Notebook)
The `Stock_Prediction_Training.ipynb` notebook contains the complete workflow for training the LSTM model.
1.  **Data Fetching**: Downloads 10 years of historical stock data for Apple (`AAPL`).
2.  **Preprocessing**: The closing prices are scaled to a range of (0, 1). Data is then transformed into sequences of 100-day windows as input (`X`) and the 101st day as the output (`y`).
3.  **Model Architecture**: A sequential LSTM model is built with Keras, using Dropout layers to prevent overfitting.
4.  **Training**: The model is trained on the preprocessed data for 50 epochs.
5.  **Export**: The final trained model is saved as `keras_model.h5`.

### 2. Web Application (Flask)
The `app.py` script serves the interactive web application.
1.  **Model Loading**: The pre-trained `keras_model.h5` is loaded into memory when the application starts.
2.  **User Input**: The user provides a stock ticker through a web form.
3.  **Live Data Processing**: The app fetches 10 years of data for the requested ticker, generates statistics, and creates visualizations for historical prices and moving averages.
4.  **Prediction**: The same preprocessing steps from training are applied to the new data, which is then fed into the LSTM model to generate a price prediction.
5.  **Display**: All charts and data are rendered on the `home.html` template.

---

## ⚠️ Important Disclaimer

The included LSTM model (`keras_model.h5`) was trained **exclusively on Apple Inc. (AAPL) stock data**. While the application allows you to input any ticker, the predictions for stocks other than AAPL are for demonstration purposes only and should **not** be considered financially accurate. The model has learned the specific patterns of AAPL's stock and will not generalize well to other securities without being retrained on their specific data.

---

## Setup and Usage

### Prerequisites
-   Python 3.8+
-   [UV](https://github.com/astral-sh/uv) (or `pip`) for package installation.

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <project-directory>
```

### 2. Set Up the Virtual Environment
We recommend using `uv` for fast dependency management.

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate
# Or on Windows
# .venv\Scripts\activate

# Install the required packages
uv pip install -r requirements.txt
```

### 3. Run the Application
Make sure `keras_model.h5` is in the root directory alongside `app.py`.

```bash
flask run
# Or
python app.py
```
The application will be available at `http://127.0.0.1:5000`.

---

## Project Structure
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