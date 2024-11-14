import yfinance as yf
from prophet import Prophet
import pandas as pd
import pickle
import plotly.graph_objects as go
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import date

app = Flask(__name__)
CORS(app)

class Model:
    def __init__(self):
        self.START = "2015-01-01"
        self.model = None

    def load_data(self, ticker, end) -> pd.DataFrame:
        data = yf.download(ticker, self.START, end)
        data.reset_index(inplace=True)
        return data

    def train(self, data, saved_model) -> bool:  # return True if new model is trained
        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        print(f"Loading saved model for {saved_model}")
        self.load_model(f'{saved_model}.pkl')
        if self.model is None:
            print(f"No saved model for {saved_model}. Training new model")
            self.model = Prophet()
            self.model.fit(df_train)
            return True
        return False

    def predict(self, period=60) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=period)
        forecast = self.model.predict(future)
        return forecast[['ds','yhat']].tail(period)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            self.model = None
            print("Model not found")


# Instantiate the model
m = Model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol').upper()
    period = int(request.args.get('period'))

    TODAY = date.today().strftime("%Y-%m-%d")
    data = m.load_data(symbol, TODAY)
    if data.empty:
        return jsonify([])

    if m.train(data, symbol):
        m.save_model(f'{symbol}.pkl')

    forecast = m.predict(period)
    forecast_list = [[x[0].strftime('%Y-%m-%d'), x[1]] for x in forecast.values.tolist()]

    # Prepare the historical data and predicted data for plotting
    historical_data = data[['Date', 'Close']]
    historical_data['Date'] = historical_data['Date'].astype(str)

    # Create Plotly graph
    fig = go.Figure()

    # Plot historical stock data
    fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Close'], mode='lines', name='Historical Price'))

    # Plot predicted stock data
    forecast_dates = [x[0].strftime('%Y-%m-%d') for x in forecast.values.tolist()]
    predicted_prices = [x[1] for x in forecast.values.tolist()]
    fig.add_trace(go.Scatter(x=forecast_dates, y=predicted_prices, mode='lines', name='Predicted Price'))

    # Customize layout
    fig.update_layout(title=f'Stock Price Prediction for {symbol}',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)',
                      template='plotly_dark')

    # Convert the figure to HTML
    graph_html = fig.to_html(full_html=False)

    return jsonify({
        'forecast': forecast_list,
        'graph': graph_html
    })


if __name__ == '__main__':
    app.run(debug=True)