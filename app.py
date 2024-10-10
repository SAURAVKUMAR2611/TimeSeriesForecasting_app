from flask import Flask, render_template, request
import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import base64
import io
import seaborn as sns


app = Flask(__name__)

stock_codes = ['84077', '85123A', '85099B', '21212', '84879', '22197', '17003', '21977', '84991', '22492']


data = pd.read_csv('Sales_Data/Final_Transactions_data.csv')  
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d-%m-%Y')

@app.route('/')
def index():
    return render_template('index.html', stock_codes=stock_codes)

@app.route('/predict', methods=['POST'])
def predict():
    stock_code = request.form['stock_code']


    stock_data = data[data['StockCode'] == stock_code]

    
    weekly_sales = stock_data.resample('W-Mon', on='InvoiceDate').sum(numeric_only=True).reset_index().sort_values(by='InvoiceDate')

    
    prophet_data = weekly_sales[['InvoiceDate', 'Quantity']].rename(columns={'InvoiceDate': 'ds', 'Quantity': 'y'})

    
    model = Prophet()
    model.fit(prophet_data)

    
    future = model.make_future_dataframe(periods=15, freq='W')
    forecast = model.predict(future)


    fig, ax = plt.subplots(figsize=(10, 6))  
    forecast_plt=forecast.loc[forecast['ds']<'2023-06-01']
    forecast_tst=forecast.loc[forecast['ds']>'2023-06-01']
    ax.plot(forecast_plt['ds'], forecast_plt['yhat'], label='Train Predicted Sales', color='red', linestyle='--')
    ax.plot(forecast_tst['ds'], forecast_tst['yhat'], label='Test Predicted Sales', color='yellow', linestyle='--')
    
    prophet_data_plt=prophet_data.loc[prophet_data['ds']<'2023-06-01']
    prophet_data_tst=prophet_data.loc[prophet_data['ds']>'2023-06-01']
    ax.plot(prophet_data_plt['ds'], prophet_data_plt['y'], label='Train Actual Sales', color='blue', marker='o')
    ax.plot(prophet_data_tst['ds'], prophet_data_tst['y'], label='Test Actual Sales', color='green', marker='o')

    
    ax.set_title(f'Prophet Forecast vs Actual Sales for Stock Code: {stock_code}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales Quantity')
    ax.axvline(x=pd.to_datetime('2023-06-01'), color='green', linestyle='--', label='Forecast Start')  # Optional line
    ax.legend(loc='best')
    ax.grid(True)

    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)  

    
    model = Prophet()
    model.fit(prophet_data)

    
    future = model.make_future_dataframe(periods=15, freq='W')
    forecast = model.predict(future)

    
    forecast_plt = forecast.loc[forecast['ds'] < '2023-06-01']
    forecast_tst = forecast.loc[forecast['ds'] >= '2023-06-01']

    train_error = prophet_data['y'] - forecast_plt['yhat']
    

    test_actual = prophet_data.loc[prophet_data['ds'] >= '2023-06-01', 'y']
    test_predicted = forecast_tst['yhat']
    test_error = test_actual.reset_index(drop=True) - test_predicted.reset_index(drop=True)

    
    fig_errors, axes = plt.subplots(1, 2, figsize=(14, 6))

    
    sns.histplot(train_error, bins=20, kde=True, color='green', ax=axes[0])
    axes[0].set_title('Training Error Distribution')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Frequency')

    
    sns.histplot(test_error, bins=20, kde=True, color='red', ax=axes[1])
    axes[1].set_title('Testing Error Distribution')
    axes[1].set_xlabel('Error')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()

    
    img_errors = io.BytesIO()
    plt.savefig(img_errors, format='png')
    img_errors.seek(0)
    error_plot_url = base64.b64encode(img_errors.getvalue()).decode()
    plt.close(fig_errors)  

    return render_template('result.html', plot_url=plot_url, error_plot_url=error_plot_url)

    

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
