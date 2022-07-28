from pathlib import Path
import pandas as pd
import numpy as np
from c_data_cleaning import *
from b_data_profile import *
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px

# df_w = pd.read_excel("E:\\D\\PHD\\term1\\visual analystic\\prj\\visual data\\temp.xlsx")
df_w = pd.read_excel("data\\temp.xlsx")

df_w['datetime'] = pd.to_datetime(df_w['datetime'], unit='ms')
df_w = df_w.melt(id_vars='datetime')
df_w.columns = ['datetime', 'city', 'temperature']

numeric_columns = get_numeric_columns(df_w)
categorical_columns = get_text_categorical_columns(df_w)

for nc in numeric_columns:
    df_w = fix_outliers(df_w, nc, OutlierAndNanFixMethod.REPLACE_MEAN)
    df_w = fix_nans(df_w, nc, OutlierAndNanFixMethod.REPLACE_MEAN)

print(df_w)
df_w['Year'] = pd.DatetimeIndex(df_w['datetime']).year
df_w['Month'] = pd.DatetimeIndex(df_w['datetime']).month
df_w['Day'] = pd.DatetimeIndex(df_w['datetime']).day

df_w = df_w.loc[df_w['Year'] > 2012].reset_index()
df_w['M_D'] = (df_w['Month'] - 1) * 30 + df_w['Day']

us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

df_w['code'] = df_w['city'].apply(lambda x: us_state_to_abbrev[x])

# fig.write_html('tmp.html', auto_open=True)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

available_indicators = df_w['Year'].unique()

app.layout = dbc.Container([
    html.H1(children='َVisual Analytics Project'),
    html.Div(
        children='This plot will show the temperature of different state of US over time'),
    html.Hr(),

    # Dataset and graph dropdowns

    dbc.FormGroup([
        dbc.Label("Choose Dataset"),
        dcc.Dropdown(id='year-column', value=2013,
                     options=[{'label': i, 'value': i} for i in available_indicators]),
    ]),

    html.Div([
        dcc.Graph(id='indicator-graphic'),
    ], style={'width': '60%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='second-graphic'),
    ], style={'width': '40%', 'float': 'right', 'display': 'inline-block'}),

    dcc.Slider(
        id='day--slider',
        min=df_w['M_D'].min(),
        max=df_w['M_D'].max(),
        value=df_w['M_D'].min(),
        step=10,
        marks={
            1: 'Winter',
            90: 'Spring',
            180: 'Summer',
            270: 'Fall'
        },
    ),
    html.Div(id='slider-output-container'),
])


@app.callback(
    Output('indicator-graphic', 'figure'),
    Output('slider-output-container', 'children'),
    Input('year-column', 'value'),
    Input('day--slider', 'value'))
def update_graph(year_column, day_value):
    sub_df = df_w.loc[df_w['Year'] == year_column]
    sub_df = sub_df[sub_df['M_D'] == day_value]
    sub_df = sub_df.sort_values('datetime')

    fig = go.Figure(data=go.Choropleth(
        locations=sub_df['code'],  # Spatial coordinates
        z=sub_df['temperature'].astype(float),  # Data to be color-coded
        locationmode='USA-states',  # set of locations match entries in `locations`
        colorscale='Reds',
        hovertext=sub_df['city'],
        colorbar_title="temperature",
    ))

    fig.update_layout(
        title_text='US temperature over time',
        geo_scope='usa',  # limite map scope to USA
    )

    text = "You have selected: " + str(sub_df['Year'].unique()[0]) + "/" + str(sub_df['Month'].unique()[0]) + "/" + str(
        sub_df['Day'].unique()[0])
    return fig, text


@app.callback(
    Output('second-graphic', 'figure'),
    Input('year-column', 'value'),
    Input('day--slider', 'value'),
    Input('indicator-graphic', 'clickData'))
def display_click_data(year_column, slider_value, click_data):
    city = click_data['points'][0]['hovertext']
    sub_df = df_w.loc[df_w['Year'] == year_column]
    sub_df = sub_df.loc[sub_df['city'] == city]
    sub_df = sub_df.sort_values('datetime')
    point_df = sub_df.loc[sub_df['M_D'] == slider_value]

    fig = go.Figure(go.Scatter(x=list(sub_df['M_D']), y=list(sub_df['temperature']), mode="lines", name='whole year'))
    fig.add_scatter(x=list(point_df['M_D']), y=list(point_df['temperature']), mode="markers", name='selected day')

    title = "Temperature of " + city + " in " + str(year_column)
    fig.update_layout(
        title=title,
        xaxis_title="day of year",
        yaxis_title="temperature")

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
