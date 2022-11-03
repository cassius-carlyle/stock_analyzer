from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from datetime import date
import csv
import datetime
from io import BytesIO
import re
import sys
import sysconfig
import urllib3

import boto3
from boto3 import client
from botocore.config import Config
import psycopg2
import pyathena
import xlwings as xw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas_datareader as pdr
import yfinance as yf
yf.pdr_override()
import plotly.graph_objs as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

default_sitepackages = r'C:\Users\chris\Documents\PYCHARM'
today=date.today()
yesterday=date.today() - datetime.timedelta(days=1)
rsi_end = yesterday
rsi_start = rsi_end - datetime.timedelta(days=360)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_athena_connection(conpath=default_sitepackages):
    with open(conpath) as f:
        lines = f.read().splitlines()
    connection = pyathena.connect(aws_access_key_id=lines[0]
                                  , aws_secret_access_key=lines[1]
                                  , s3_staging_dir=lines[2]
                                  , region_name=lines[3])
    return connection

def get_s3_info(bucket='data-dump12', conpath=f'{default_sitepackages}\.s3con.txt'):
    """
    Constructor for other methods that require S3 authentication

    It is unlikely that you will need to invoke this method directly

    :param bucket: S3 bucket for info to be authenticated against
    :type bucket: str, default 'dish-prod-sbx-slnganalytics'
    :param conpath: Local storage filepath to a .txt file with lines for `access key`, `secret key`
    :type conpath: str, default ...site-packages/aws_methods/.s3con.txt
    :return: Key/value pairs useful for authenticating AWS credentials.
    :rtype: dict
    """
    s3_info = {'bucket': bucket}
    with open(conpath, 'r') as f:
        lines = f.read().splitlines()
        s3_info['access_key'] = lines[0]
        s3_info['secret'] = lines[1]
    return s3_info

def get_s3_client(  # config=None,
        **kwargs):
    """
    Basic helper wrapper for any function needing to operate on the S3 file system/metadata.

    See :py:func:`~aws_methods.aws_methods.get_s3_info` for a list of keyword arguments and defaults.

    :param config: Credentials to a proxy connection if required
    :type config: :class:`botocore.config.Config`, default None
    :return: A low-level authenticated S3 client resource
    :rtype: S3 client

    :Example:

        .. code-block:: python

            s3_clt = aws_methods.get_s3_client()
    """
    s3_info = get_s3_info(**kwargs)
    client = boto3.client('s3'
                          , region_name='us-west-2'
                          , verify=False
                          , aws_access_key_id=s3_info['access_key']
                          , aws_secret_access_key=s3_info['secret']
                          # , config=config
                          )
    return client

def upload_df(df, key, bucket='data-dump12', s3_client=get_s3_client(),  # config=None,
              **kwargs):
    """
    Upload a pandas dataframe to a new S3 file via memory stream

    :param df: Data to be uploaded
    :type df: :class:`pandas.DataFrame`
    :param key: File name/location for uploaded data within S3 `bucket`
    :type key: str
    :param bucket: S3 bucket to store `key` in
    :type bucket: str, default 'dish-prod-sbx-slnganalytics'
    :param s3_client: A low-level authenticated S3 client resource
    :type s3_client: S3 client, default :py:func:`aws_methods.aws_methods.get_s3_client()`

    :Example:

        .. code-block:: python

            mydf = pd.DataFrame({'column_name': [1,2,3,4,5]})
            # upload as parquet
            aws_methods.upload_df(
                mydf
                , 'ec2-user/dummydata/mydata.parquet')
            # upload as csv
            aws_methods.upload_df(
                mydf
                , 'ec2-user/dummydata/mydata.csv')
    """
    upload_filetype = re.sub('^\.', '', os.path.splitext(key)[1])
    out_buffer = BytesIO()
    if upload_filetype in ['parquet', 'snappy', 'gz']:
        compression = upload_filetype if upload_filetype == 'gz' else 'snappy'
        df.to_parquet(out_buffer, compression=compression, index=False, **kwargs)
    elif upload_filetype in ['csv', 'txt']:
        df.to_csv(out_buffer, index=False, **kwargs)
    elif upload_filetype in ['xls', 'xlsx']:
        df.to_excel(out_buffer, index=False, **kwargs)
    else:
        raise ValueError(
            'Acceptable file extensions for "key" include: ("parquet.*", "txt", "csv", "excel"). Please specify one of those.')
    s3_client.put_object(Bucket=bucket, Key=key, Body=out_buffer.getvalue()  # , config=config
                         )

def Ticker(tickers, start, end):
    """
    Return historical price data for a list of stock tickers over specified period of time

    :param tickers: list of tickers
    :type tickers: list, str
    :param start: start of fetching period
    :type start: date
    :param end: end of fetching period
    :type end: date

    """
    df = pd.DataFrame(columns=['Ticker', 'Open', 'Low', 'High', 'Close', 'Adj Close', 'Volume', 'Date'])
    for ticker in tickers:
        print(ticker)
        data = yf.download(tickers=ticker, start=start, end_date=end, threads=True)  # period='1mo'
        data['Ticker'] = str(ticker)
        data = data.reset_index()
        data = data[['Ticker', 'Open', 'Low', 'High', 'Close', 'Adj Close', 'Volume', 'Date']]
        df = pd.concat((df, data), axis=0)

    return df

def bollinger_bands_plot(ticker_df, window):
    """
    Return a plot of a stock or weighted portfolio of stocks and the resepctive bollenger bands

    :param ticker_df: The resulting dataframe from the 'Ticker' function
    :type ticker_df: dataframe
    :param window: the timeframe used to calculate the SMA, STD and resulting bands
    :type window: Integer

    """
    closing_prices = ticker_df['Adj Close']
    symbol = ticker_df['Ticker'].unique()
    sma = closing_prices.rolling(window).mean() # <-- Get SMA for 20 days
    std = closing_prices.rolling(window).std() # <-- Get rolling standard deviation for 20 days
    top_band = sma + std * 2 # Calculate top band
    bottom_band = sma - std * 2 # Calculate bottom band

    plt.title(symbol + ' Bollinger Bands')
    plt.xlabel('Days')
    plt.ylabel('Closing Prices')
    plt.plot(closing_prices, label='Closing Prices')
    plt.plot(top_band, label='Bollinger Up', c='g')
    plt.plot(bottom_band, label='Bollinger Down', c='r')
    plt.legend()
    plot = plt.show()
    return plot

def bollinger_bands_dash(ticker_df, window):
    """
    Return a plot of a stock or weighted portfolio of stocks and the resepctive bollenger bands

    :param ticker_df: The resulting dataframe from the 'Ticker' function
    :type tickers_df: dataframe
    :param window: the timeframe used to calculate the SMA, STD and resulting bands
    :type window: Integer

    """
    closing_prices = ticker_df['Adj Close']
    dates = ticker_df['Date']
    symbol = ticker_df['Ticker'].unique()
    sma = closing_prices.rolling(window).mean() # <-- Get SMA for 20 days
    std = closing_prices.rolling(window).std() # <-- Get rolling standard deviation for 20 days
    top_band = sma + std * 2 # Calculate top band
    bottom_band = sma - std * 2 # Calculate bottom band

    bands_df = ticker_df[['Ticker', 'Adj Close', 'Date']]
    bands_df['SMA'] = sma
    bands_df['STD'] = std
    bands_df['Top Band'] = top_band
    bands_df['Low Band'] = bottom_band
    bands_df = bands_df.iloc[20:]
    return bands_df

def metadata(tickers):
    """
    Return metadata about a specific stock, suc has PE, Beta etc.

    :param tickers: list of tickers
    :type tickers: list, str

    """
    df = pd.DataFrame(columns=['sector','country','industry','profitMargins','grossMargins','revenueGrowth','targetLowPrice'
        ,'targetMedianPrice','earningsGrowth','returnOnAssets','debtToEquity','quickRatio','priceToBook','beta','forwardPE'
        ,'trailingPE','shortPercentOfFloat','marketCap','averageVolume','volume','dividendYield','Ticker','Date'])

    for ticker in tickers:
        sym = yf.Ticker(ticker)
        meta_dict = sym.info
        meta_dict_cut = dict((k, meta_dict[k]) for k in ['sector', 'country','industry','profitMargins','grossMargins','revenueGrowth'
        ,'targetLowPrice','targetMedianPrice','earningsGrowth','returnOnAssets','debtToEquity','quickRatio'
        ,'priceToBook','beta','forwardPE','trailingPE','shortPercentOfFloat','marketCap','averageVolume','volume','dividendYield']
               if k in meta_dict)
        meta_dict_cut['Ticker'] = ticker
        meta_dict_cut['Date'] = today
        meta_df = pd.DataFrame([meta_dict_cut])
        df = pd.concat((df, meta_df), axis=0)
        sector = df['sector'].unique()
        industry = df['industry'].unique()
        beta = df['beta'].unique()
        PE = df['trailingPE'].unique()
        marketCap = df['marketCap'].unique()

        sector = sector[0]
        industry = industry[0]
        beta = beta[0]
        PE = PE[0]
        marketCap = f"${marketCap[0]:,}"

    return sector, industry, beta, PE, marketCap

def get_plot(tickers, start, end, window=20, plot_lookback=365):
    """
    Return historical price data for a list of stock tickers over specified period of time

    :param tickers: list of tickers
    :type tickers: list, str
    :param start: start of fetching period
    :type start: date
    :param end: end of fetching period
    :type end: date
    :param window (optional - default 20): the timeframe used to calculate the SMA, STD and resulting bands
    :type window: Integer
    :param plot_lookback (optional - default 365 days): how many days the plot should look back
    :type plot_lookback: Integer

    """
    df = pd.DataFrame(columns=['Ticker', 'Open', 'Low', 'High', 'Close', 'Adj Close', 'Volume', 'Date'])
    for ticker in tickers:
        print(ticker)
        data = yf.download(tickers=ticker, start=start, end_date=end, threads=True)  # period='1mo'
        data['Ticker'] = str(ticker)
        data = data.reset_index()
        data = data[['Ticker', 'Open', 'Low', 'High', 'Close', 'Adj Close', 'Volume', 'Date']]
        df = pd.concat((df, data), axis=0)

        ticker_df = df
        closing_prices = ticker_df['Adj Close']
        dates = ticker_df['Date']
        sma = closing_prices.rolling(window).mean()  # <-- Get SMA for 20 days
        std = closing_prices.rolling(window).std()  # <-- Get rolling standard deviation for 20 days
        top_band = sma + std * 2  # Calculate top band
        bottom_band = sma - std * 2  # Calculate bottom band

        ticker_slice = ticker_df[['Ticker', 'Adj Close', 'Date']]
        ticker_slice['SMA'] = sma
        ticker_slice['STD'] = std
        ticker_slice['Top Band'] = top_band
        ticker_slice['Low Band'] = bottom_band
        ticker_slice = ticker_slice.tail(plot_lookback)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_slice['Date'], y=ticker_slice['Adj Close'], name='Adj Close',
                                 line=dict(color='white', width=3)))

        fig.add_trace(go.Scatter(x=ticker_slice['Date'], y=ticker_slice['Top Band'], name='Top Band',
                                 line=dict(color='red', width=3)))

        fig.add_trace(go.Scatter(x=ticker_slice['Date'], y=ticker_slice['Low Band'], name='Low Band',
                                 line=dict(color='green', width=3)))

        fig.update_layout(title=str(ticker),title_x=0.5, xaxis_title='Date', yaxis_title='Price')

        fig.update_layout(template='plotly_dark', titlefont=dict(family="Times New Roman", size=24))

    return fig

def get_doubles(tickers, start, end, double_range = 0.01):

    """
    Return historical price data for a list of stock tickers over specified period of time

    :param tickers: list of tickers
    :type tickers: list, str
    :param start: start of fetching period
    :type start: date
    :param end: end of fetching period
    :type end: date
    :param double_range: % the most recent price can be within another bottom/top to be triggered as a DB or DT
    :type end: decimal e.g. 0.01

    """
    for ticker in tickers:
        ticker_df = Ticker(tickers, start, end)
        print(ticker_df)
        print(ticker_df.last_valid_index())
        X = ticker_df['Date']
        Y = ticker_df['Adj Close']
        x_data = ticker_df.index.tolist()
        y_data = ticker_df['Low']
        y_data2 = ticker_df['High']
        x = np.linspace(0, max(ticker_df.index.tolist()), max(ticker_df.index.tolist()) + 1)

        # # polynomial fit of degree xx
        pol_g = np.polyfit(x_data, Y, 80)
        y_pol_g = np.polyval(pol_g, x)

        pol = np.polyfit(x_data, y_data, 90)
        y_pol = np.polyval(pol, x)
        pol2 = np.polyfit(x_data, y_data2, 90)
        y_pol2 = np.polyval(pol2, x)

        ticker_df['Low'] = ticker_df['Low'].astype(float)
        data = ticker_df['Low']
        # data = y_pol
        # data2 = y_pol2
        ticker_df['High'] = ticker_df['High'].astype(float)
        data2 = ticker_df['High']

        l_min = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1  # local min
        l_min = np.append(l_min, ticker_df.last_valid_index())

        l_max2 = (np.diff(np.sign(np.diff(data2))) < 0).nonzero()[0] + 1  # local max
        l_max2 = np.append(l_max2, ticker_df.last_valid_index())

        delta = 0
        df_len = len(ticker_df.index)
        ################################
        ##### DOUBLE BOTTOMS START #####
        ################################
        dict_i = dict()
        dict_x = dict()
        for element in l_min:
            l_bound = element - delta
            u_bound = element + delta
            x_range = range(l_bound, u_bound + 1)  # range of x positions where we SUSPECT to find a low
            dict_x[element] = x_range  ##Why does dict_i have three values, we only need the low

            y_loc_list = list()
            for x_element in x_range:
                if x_element > 0 and x_element < df_len:  # need to stay within the dataframe
                    y_loc_list.append(
                        ticker_df['Low'].iloc[x_element])  # list of suspected y values that can be a minimum
            dict_i[element] = y_loc_list
        # print(l_min)
        # print(dict_i)
        y_delta = 0.005  # percentage distance between average lows
        threshold = min(ticker_df['Low']) * 1.20  # setting threshold higher than the global low


        y_dict = dict()
        mini = list()
        suspected_bottoms = list()
        for key in dict_i.keys():  # for suspected minimum x position
            mn = sum(dict_i[key]) / len(dict_i[key])  # this is averaging out the price around that suspected minimum

            price_min = min(dict_i[key])
            mini.append(price_min)  # lowest value for price around suspected

            l_y = mn * (1.0 - y_delta)
            u_y = mn * (1.0 + y_delta)
            y_dict[key] = [l_y, u_y, mn, price_min]
        print(y_dict)
        for key_i in y_dict.keys():
            for key_j in y_dict.keys():
                if (key_i != key_j) and (y_dict[key_i][3] < threshold): #Either use Low or Avg to compare to peers
                    if (y_dict[key_i][3] < y_dict[key_j][3]*(1+double_range)) and (y_dict[key_i][3] > y_dict[key_j][3]*(1-double_range)):
                        if (key_i == ticker_df.last_valid_index()) or (key_j == ticker_df.last_valid_index()):
                            suspected_bottoms.append(key_i)
                            print('----------------------- ')
                            print('--- Bottoming pattern found for x index pair: ', key_i, ',', key_j)
                            print('----------------------- ')
                else:
                    pass
        suspected_bottoms = [*set(suspected_bottoms)] #dedup list

        values = list()
        for i in range(len(suspected_bottoms)):
            value = suspected_bottoms[i]
            value = ticker_df['Date'].iloc[value]
            value = value.strftime('%Y-%m-%d')
            values.append(value)
        # print(suspected_bottoms)
        # print(values)
        ################################
        ###### DOUBLE BOTTOMS END ######
        ################################
        print('#############################################################')
        ################################
        ####### DOUBLE TOPS START ######
        ################################
        dict_i2 = dict()
        dict_x2 = dict()
        for element in l_max2:
            l_bound = element - delta
            u_bound = element + delta
            x_range2 = range(l_bound, u_bound + 1)  # range of x positions where we SUSPECT to find a high
            dict_x2[element] = x_range2

            y_loc_list2 = list()
            for x_element in x_range2:
                if x_element > 0 and x_element < df_len:
                    y_loc_list2.append(
                        ticker_df['High'].iloc[x_element])  # list of suspected y values that can be a maximum
            dict_i2[element] = y_loc_list2
        # print(l_max2)
        # print(dict_i2)

        y_delta = 0.005  # percentage distance between average lows
        threshold2 = max(ticker_df['High']) * 0.80  # setting threshold higher than the global high

        y_dict2 = dict()
        maxi = list()
        suspected_tops = list()
        #   BUG somewhere here
        for key in dict_i2.keys():  # for suspected minimum x position
            mn = sum(dict_i2[key]) / len(dict_i2[key])
            price_max = max(dict_i2[key])
            maxi.append(price_max)  # lowest value for price around suspected

            l_y = mn * (1.0 - y_delta)
            u_y = mn * (1.0 + y_delta)
            y_dict2[key] = [l_y, u_y, mn, price_max]
        # print(y_dict2)

        for key_i in y_dict2.keys():
            for key_j in y_dict2.keys():
                if (key_i != key_j) and (y_dict2[key_i][3] > threshold2):
                    if (y_dict2[key_i][3] < y_dict2[key_j][3]*(1+double_range)) and (y_dict2[key_i][3] > y_dict2[key_j][3]*(1-double_range)):
                        if (key_i == ticker_df.last_valid_index()) or (key_j == ticker_df.last_valid_index()):
                            print('----------------------- ')
                            print('--- Topping pattern found for x index pair: ', key_i, ',', key_j)
                            suspected_tops.append(key_i)
                            print('----------------------- ')
                    else:
                        pass
        suspected_tops = [*set(suspected_tops)]  # dedup list

        values2 = list()
        for i in range(len(suspected_tops)):
            value = suspected_tops[i]
            value = ticker_df['Date'].iloc[value]
            value = value.strftime('%Y-%m-%d')
            values2.append(value)
        # print(suspected_tops)
        # print(values2)
        ################################
        ######## DOUBLE TOPS END #######
        ################################
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Adj Close'], name='Adj Close', opacity=0.5,
                                 line=dict(color='white', width=1)))

        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['High'], name='High', mode='markers',
                                 marker_color='red', marker_symbol='circle', marker_size=5))

        # fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Open'], name='Open', mode='markers',
        #                          marker_color='grey', marker_symbol='circle', marker_size=3))

        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Low'], name='Low', mode='markers',
                                 marker_color='green', marker_symbol='circle', marker_size=5))

        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=y_pol, name='Polyfit Bottom', opacity=0.7,
                                 line=dict(color='#53d21d', width=3)))

        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=y_pol2, name='Polyfit Top', opacity=0.7,
                                 line=dict(color='#cc0000', width=3)))

        for position in values:
            fig.add_vrect(x0=position, x1=position, line_dash="dot", line_color='green'
                          , annotation_text="DB", opacity=0.9)

        for position in values2:
            fig.add_vrect(x0=position, x1=position, line_dash="dot", line_color='red'
                          , annotation_text="DT", opacity=0.9)

        fig.update_layout(title=str(ticker), title_x=0.5, xaxis_title='Date', yaxis_title='Price')

        fig.update_layout(template='plotly_dark', titlefont=dict(family="Times New Roman", size=24))
    return fig

def get_rsi(tickers, start,end):
    """
    Return historical price data for a list of stock tickers over specified period of time

    :param tickers: list of tickers
    :type tickers: list, str
    :param start: start of fetching period
    :type start: date
    :param end: end of fetching period
    :type end: date

    """
    delta = end - start
    look_back = delta.days
    for ticker in tickers:
        ticker_df = Ticker(tickers, start, end)
        X = ticker_df['Date']
        Y = ticker_df['Adj Close']

        ########  RSI  ##########
        def rsi_calc(data, time_window):
            diff = Y.diff(1).dropna()  # diff in one field(one day)

            up_chg = 0 * diff
            down_chg = 0 * diff

            # up change is equal to the positive difference, otherwise equal to zero
            up_chg[diff > 0] = diff[diff > 0]

            # down change is equal to negative deifference, otherwise equal to zero
            down_chg[diff < 0] = diff[diff < 0]

            up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
            down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

            rs = abs(up_chg_avg / down_chg_avg)
            rsi = 100 - 100 / (1 + rs)
            return rsi

        ticker_df['RSI'] = rsi_calc(Y,14)
        # print(ticker_df)

        ###########  STOCHASTIC  ###########
        def stochastic_calc(data, k_window, d_window, window):
            min_val = Y.rolling(window=window, center=False).min()
            max_val = Y.rolling(window=window, center=False).max()

            stoch = ((Y - min_val) / (max_val - min_val)) * 100

            K = stoch.rolling(window=k_window, center=False).mean()
            D = K.rolling(window=d_window, center=False).mean()
            return K, D

        ticker_df['K'], ticker_df['D'] = stochastic_calc(ticker_df['RSI'], 3, 3, 14)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=ticker_df['Date'].tail(60), y=ticker_df['K'].tail(look_back), name='STOCH RSI K', opacity=0.9,
                                 line=dict(color='blue', width=3)))

        fig.add_trace(go.Scatter(x=ticker_df['Date'].tail(60), y=ticker_df['D'].tail(look_back), name='STOCH RSI D', opacity=0.6,
                                 line=dict(color='green', width=3)))

        # fig.add_trace(go.Scatter(x=ticker_df['Date'].tail(60), y=ticker_df['RSI'].tail(60), name='STOCH RSI D', opacity=0.9,
        #                          line=dict(color='orange', width=3)))

        fig.add_hrect(y0=20, y1=20, line_dash="dot", line_color='green'
                      , annotation_text="Over Sold", opacity=0.9)

        fig.add_hrect(y0=80, y1=80, line_dash="dot", line_color='red'
                      , annotation_text="Over Bought", opacity=0.9)

        fig.add_hrect(y0=100, y1=100, line_dash="dot", line_color='red'
                      , annotation_text="Very Over Bought", opacity=0.5)

        fig.add_hrect(y0=0, y1=0, line_dash="dot", line_color='green'
                      , annotation_text="Very Over Sold", opacity=0.5)

        fig.update_layout(title=str(ticker) + " - Stochastic RSI", title_x=0.5, xaxis_title='Date', yaxis_title='RSI')

        fig.update_layout(template='plotly_dark', titlefont=dict(family="Times New Roman", size=24))

    return fig
