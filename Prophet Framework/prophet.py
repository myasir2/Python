# -*- coding: utf-8 -*-

from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, plot
from plotly import graph_objs as go

import logging
logging.getLogger().setLevel(logging.ERROR)
 
data = pd.read_csv('./medium_posts.csv', sep = '\t')
data = data[['published', 'url']].dropna().drop_duplicates()
data['published'] = pd.to_datetime(data['published'])
data.sort_values(by = ['published']).head(n = 3)
data = data[(data['published'] > '2012-08-15') & 
            (data['published'] < '2017-06-26')].sort_values(by = ['published'])
data.head(n = 3)

# Aggregate data into unique posts
aggregate_data = data.groupby('published')[['url']].count()
aggregate_data.columns = ['posts']

# Resample data into 1-day bins
daily_data = aggregate_data.resample('D').apply(sum)
daily_data = daily_data.loc[daily_data.index >= '2015-01-01']
plotly_df(daily_data, title = 'Posts on Medium(daily)')

# Resample data into weekly bins
weekly_data = daily_data.resample('W').apply(sum)
plotly_df(weekly_data, title = 'Posts on Medium(weekly)')

# Prophet data pre-processing
data = daily_data.reset_index()
data.columns = ['ds', 'y']
data.tail()

# Train/Future split
train = data[: -30]
prophet = Prophet()
prophet.fit(train)
future = prophet.make_future_dataframe(periods = 30)

# Forecast
forecast = prophet.predict(future)
prophet.plot(forecast)
prophet.plot_components(forecast)

# Evaluation
evaluation = make_comparison_dataframe(data, forecast)
for err_name, err_value in calculate_forecast_errors(evaluation, 30).items():
    print(err_name, err_value)
show_forecast(evaluation, 30, 100, 'New posts on Medium')

# Box-Cox transformation
train2 = train.copy()
train2['y'], lambda_prophet = stats.boxcox(train2['y'])

# Initialize new Prophet forecast
prophet2 = Prophet()
prophet2.fit(train2)
future2 = prophet2.make_future_dataframe(periods = 30)
forecast2 = prophet2.predict(future2)

prophet2.plot(forecast)

# Inverse values created by the Box-Cox transformation
for column in ['yhat', 'yhat_lower', 'yhat_upper']:
    forecast2[column] = inverse_boxcox(forecast2[column], lambda_prophet)

# Evaluation
evaluation2 = make_comparison_dataframe(data, forecast2)
for err_name, err_value in calculate_forecast_errors(evaluation2, 30).items():
    print(err_name, err_value)
show_forecast(evaluation2, 30, 100, 'New posts on Medium')

def plotly_df(df, title=''):
    """Visualize all the dataframe columns as line plots."""
    common_kw = dict(x=df.index, mode='lines')
    data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]
    layout = dict(title=title)
    fig = dict(data=data, layout=layout)
    plot(fig, show_link=False)
    
def make_comparison_dataframe(historical, forecast):
    """Join the history with the forecast.
    
       The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.
    """
    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))

def calculate_forecast_errors(df, prediction_size):
    """Calculate MAPE and MAE of the forecast.
    
       Args:
           df: joined dataset with 'y' and 'yhat' columns.
           prediction_size: number of days at the end to predict.
    """
    
    # Make a copy
    df = df.copy()
    
    # Now we calculate the values of e_i and p_i according to the formulas given in the article above.
    df['e'] = df['y'] - df['yhat']
    df['p'] = 100 * df['e'] / df['y']
    
    # Recall that we held out the values of the last `prediction_size` days
    # in order to predict them and measure the quality of the model. 
    
    # Now cut out the part of the data which we made our prediction for.
    predicted_part = df[-prediction_size:]
    
    # Define the function that averages absolute error values over the predicted part.
    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))
    
    # Now we can calculate MAPE and MAE and return the resulting dictionary of errors.
    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}

def show_forecast(cmp_df, num_predictions, num_values, title):
    """Visualize the forecast."""
    
    def create_go(name, column, num, **kwargs):
        points = cmp_df.tail(num)
        args = dict(name=name, x=points.index, y=points[column], mode='lines')
        args.update(kwargs)
        return go.Scatter(**args)
    
    lower_bound = create_go('Lower Bound', 'yhat_lower', num_predictions,
                            line=dict(width=0),
                            marker=dict(color="red"))
    upper_bound = create_go('Upper Bound', 'yhat_upper', num_predictions,
                            line=dict(width=0),
                            marker=dict(color="red"),
                            fillcolor='rgba(68, 68, 68, 0.3)', 
                            fill='tonexty')
    forecast = create_go('Forecast', 'yhat', num_predictions,
                         line=dict(color='rgb(31, 119, 180)'))
    actual = create_go('Actual', 'y', num_values,
                       marker=dict(color="red"))
    
    # In this case the order of the series is important because of the filling
    data = [lower_bound, upper_bound, forecast, actual]

    layout = go.Layout(yaxis=dict(title='Posts'), title=title, showlegend = False)
    fig = go.Figure(data=data, layout=layout)
    plot(fig, show_link=False)

def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)
