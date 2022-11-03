import time
import os
from datetime import date
import csv
import datetime

import model_pipeline #Data Wrangling Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

######################################################################
###############      VARIABLE SETTING    #############################
######################################################################

today=date.today().strftime('%B %d, %Y')
yesterday=date.today() - datetime.timedelta(days=1)
end_date = date.today()
start_date = end_date - datetime.timedelta(days=60)

######################################################################
###############       DASH CALLBACKS     #############################
######################################################################

app = dash.Dash(__name__, )
#BOLLINGER BANDS
@app.callback(
    [Output(component_id='BB', component_property='figure')],
    [Input(component_id='button',component_property='n_clicks'),
    State(component_id='ticker-input', component_property='value')],
)
def bb_graph(n_clicks, value):
    if value == None:
        tick = ['AAPL']
    else:
        tick = [value]
    fig = model_pipeline.get_plot(tick, start_date, end_date, plot_lookback=90, window=10)
    return [fig]

#DOUBLE BOTTOMS_TOPS
@app.callback(
    [Output(component_id='DBT', component_property='figure')],
    [Input(component_id='button',component_property='n_clicks'),
    State(component_id='ticker-input', component_property='value')],
)
def dbt_graph(n_clicks, value):
    if value == None:
        tick = ['AAPL']
    else:
        tick = [value]
    fig = model_pipeline.get_doubles(tick,start_date,end_date)
    return [fig]

#STOCHASTIC RSI
@app.callback(
    [Output(component_id='RSI', component_property='figure')],
    [Input(component_id='button',component_property='n_clicks'),
    State(component_id='ticker-input', component_property='value')],
)
def rsi_graph(n_clicks, value):
    if value == None:
        tick = ['AAPL']
    else:
        tick = [value]
    fig = model_pipeline.get_rsi(tick, start_date, end_date)
    return [fig]

#META INFO
@app.callback(
    [Output(component_id='sector', component_property='children'),
    Output(component_id='industry', component_property='children'),
    Output(component_id='beta', component_property='children'),
    Output(component_id='pe', component_property='children'),
    Output(component_id='marketcap', component_property='children'),
     ],
    [Input(component_id='button',component_property='n_clicks'),
    State(component_id='ticker-input', component_property='value')],
)
def meta_info(n_clicks, value):
    if value == None:
        tick = ['AAPL']
    else:
        tick = [value]
    sector = model_pipeline.metadata(tick)[0]
    industry = model_pipeline.metadata(tick)[1]
    beta = model_pipeline.metadata(tick)[2]
    pe = model_pipeline.metadata(tick)[3]
    marketcap = model_pipeline.metadata(tick)[4]
    return sector, industry, beta, pe, marketcap

######################################################################
###############        DASH LAYOUT       #############################
######################################################################

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('logo2.JPG'), id = 'logo-image'
                     , style={'height': '200px', 'width': 'auto', 'margin-bottom': '25px', 'fontSize': 30})
        ], className='one-third column'),

        html.Div([
            html.Div([
                html.H3('Stock Analyzer'
                        , style={'textAlign': 'center','margin-bottom': '0px', 'color':'white', 'font-family': "Times New Roman"}),
                html.H5('Risk and Return Analysis'
                        , style={'textAlign': 'center','margin-bottom': '0px', 'color':'white', 'font-family': "Times New Roman"})
            ])
        ], className='one-hlaf column', id= 'title'),

        html.Div([
            html.H6('Updated As Of: ' + str(today), style={'color': 'orange', 'font-family': "Times New Roman"})

        ], className='one-third column', id='title1')

    ], id = 'header', className= 'row flex-display', style={'margin-bottom': '25px'}),

    html.Div([
        dcc.Input(id='ticker-input', type='text'
                  , style={'font-style':'italic','textAlign': 'center'}),
        html.Button('Fetch Stock Info', id='button', style={'color':'orange'}),
        html.Div(id='my-div')
    ]),

    html.Div([
        html.Div([
            html.H3(children='Sector:'
                    , style={'textAlign': 'center', 'margin-bottom': '0px', 'color': 'white', 'font-family': "Times New Roman"
                    , 'fontSize': 30, 'margin-top':'-10px'}),
            html.P(id='sector',
                   style={'textAlign': 'center', 'color': 'orange', 'font-family': "Times New Roman", 'fontSize': 20
                       , 'margin-bottom': '-10px', 'font-style': 'italic'})
        ], className='card_container two columns', style={'margin-bottom': '25px'}),

        html.Div([
            html.H3(children='Industry:'
                    , style={'textAlign': 'center', 'margin-bottom': '0px', 'color': 'white',
                             'font-family': "Times New Roman"
                    , 'fontSize': 30, 'margin-top': '-10px'}),
            html.P(id='industry',
                   style={'textAlign': 'center', 'color': 'orange', 'font-family': "Times New Roman", 'fontSize': 20
                       , 'margin-bottom': '-10px', 'font-style': 'italic'})
        ], className='card_container two columns', style={'margin-bottom': '25px'}),

        html.Div([
            html.H3(children='Beta:'
                    , style={'textAlign': 'center', 'margin-bottom': '0px', 'color': 'white',
                             'font-family': "Times New Roman"
                    , 'fontSize': 30, 'margin-top': '-10px'}),
            html.P(id='beta',
                   style={'textAlign': 'center', 'color': 'orange', 'font-family': "Times New Roman", 'fontSize': 20
                       , 'margin-bottom': '-10px', 'font-style': 'italic'})
        ], className='card_container two columns', style={'margin-bottom': '25px'}),

        html.Div([
            html.H3(children='PE:'
                    , style={'textAlign': 'center', 'margin-bottom': '0px', 'color': 'white',
                             'font-family': "Times New Roman"
                    , 'fontSize': 30, 'margin-top': '-10px'}),
            html.P(id='pe',
                   style={'textAlign': 'center', 'color': 'orange', 'font-family': "Times New Roman", 'fontSize': 20
                       , 'margin-bottom': '-10px', 'font-style': 'italic'})
        ], className='card_container two columns', style={'margin-bottom': '25px'}),

        html.Div([
            html.H3(children='Market Cap:'
                    , style={'textAlign': 'center', 'margin-bottom': '0px', 'color': 'white',
                             'font-family': "Times New Roman"
                    , 'fontSize': 30, 'margin-top': '-10px'}),
            # html.P(f"${id='marketcap':,}",
            html.P(id='marketcap',
                   style={'textAlign': 'center', 'color': 'orange', 'font-family': "Times New Roman", 'fontSize': 20
                       , 'margin-bottom': '-10px', 'font-style': 'italic'})
        ], className='card_container two columns', style={'margin-bottom': '25px'})

    ], className='row flex display'),

    html.Div([
        html.Div([
            html.H6(children='Bollinger Bands'
                    , style={'textAlign':'center', 'color':'white', 'font-family': "Times New Roman", 'fontSize': 20
                             ,'margin-top': '-10px'}),
            html.Div([
                dcc.Graph(id='BB')
            ])
        ], className='card_container eleven columns'),


    ], className='row flex display'),

    html.Div([
        html.Div([
            html.H6(children='Double Bottoms & Tops'
                    , style={'textAlign': 'center', 'color': 'white', 'font-family': "Times New Roman", 'fontSize': 20
                    , 'margin-top': '-10px'}),
            html.H5(children='Only shows up if today is hitting one'
                    , style={'textAlign': 'center', 'color': 'grey', 'font-family': "Times New Roman", 'fontSize': 15
                    , 'margin-top': '-10px'}),

            html.Div([
                dcc.Graph(id='DBT')
            ])
        ], className='card_container eleven columns')

    ], className='row flex display'),

    html.Div([
        html.Div([
            html.H6(children='Stochastic RSI'
                    , style={'textAlign': 'center', 'color': 'white', 'font-family': "Times New Roman", 'fontSize': 20
                    , 'margin-top': '-10px'}),
            html.H5(children='Max lookback is 360 days'
                    , style={'textAlign': 'center', 'color': 'grey', 'font-family': "Times New Roman", 'fontSize': 15
                    , 'margin-top': '-10px'}),

            html.Div([
                dcc.Graph(id='RSI')
            ])
        ], className='card_container eleven columns')

    ], className='row flex display')

], id = 'mainContainer', style={'display': 'flex', 'flex-direction': 'column'})


if __name__ == '__main__':
    app.run_server(debug=True)