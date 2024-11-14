# Stock Price Prediction App
This is a Flask app that predicts stock prices using the Prophet library from Facebook.

## Getting Started
These instructions will help you set up and run the app locally.

### Prerequisites
Before you can run the app, you need to make sure that you have the following software installed:

- Python 3.6 or higher
- Flask
- Prophet
- Pandas
- yfinance
You can install these packages by running the following command:

```bash
pip install -r requirements.txt
```

### Running the App
To run the app, you can visit: <br>
<a href="https://hardageri.github.io/Web-Technology-Project/" target="_blank">stock-prediction</a>

Source code for frontend: <br>
<a href="https://github.com/Hardageri/Web-Technology-Project" target="_blank">stock-prediction-frontend</a>

**OR**

you can execute the following command in your terminal:

```bash
python app.py
```

By default, the app will run on port 5000.

### Making a Prediction
You can make a prediction by calling the **`/predict`** endpoint of the app with the following parameters:

- **`symbol`**: The stock symbol that you want to predict.
- **`period`**: The number of days into the future that you want to predict.

For example, to predict the stock price of Apple (AAPL) for the next 30 days, you can make the following request:

```
http://localhost:5000/predict?symbol=AAPL&period=30
```

The response will be a JSON object containing the predicted stock prices for the next 30 days.

## Acknowledgments
- [Prophet library from Facebook](https://facebook.github.io/prophet/)
- [yfinance library](https://pypi.org/project/yfinance/)