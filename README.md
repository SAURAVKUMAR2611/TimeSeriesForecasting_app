# Time Series Forecasting for Sales Application

#### This application forecasts sales quantities for selected stock codes using a Flask-based backend. It leverages the Facebook Prophet model for time series prediction and provides interactive visualizations to compare actual sales with predicted values.

## Features
* Interactive web interface for selecting stock codes and generating forecasts.
* Forecasting model trained on weekly sales data.
* Visualization of predicted vs. actual sales and error distributions for both training and testing phases.
* Customizable forecasting horizon.
* Histogram plots for training and testing error distributions.

## Installation
### Prerequisites
1. Python 3.8 or higher
2. Libraries:
  * Flask
  * pandas
  * prophet
  * matplotlib
  * seaborn

Could you make sure pip is installed to handle Python dependencies?

## Steps
1. Clone the repository:

   git clone \<repository\-link\>
   
    cd \<repository\-folder\>

2. Set up the Python environment:

   python3 -m venv venv

   source venv/bin/activate  # For Linux/MacOS

   venv\Scripts\activate     # For Windows

3. Install required packages:

     pip install \-r requirements.txt


5. Prepare the sales data:

    python app.py

6. Run the application: Open your web browser and navigate to http://127.0.0.1:5000.

### Usage
#### Home Page:
* Displays a dropdown menu with available stock codes for forecasting.
#### 2.Forecast:
* Select a stock code and click Submit.
* The app generates:
    * Forecast plots (train/test splits and predicted values).
    * Error distribution histograms.
#### Output:
  * Time series plot: Shows actual vs. predicted sales quantities.
  * Histograms: Provide insights into the distribution of training and testing errors.

### Dependencies
* Install dependencies via requirements.txt:
* Flask
* pandas
* prophet
* matplotlib
* seaborn

### Acknowledgments
* Prophet by Facebook: Used for time series forecasting.
* Flask: Lightweight web framework.
* Matplotlib & Seaborn: For visualizations.




