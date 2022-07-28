import os
from typing import Dict

from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
import plotly.figure_factory as ff

from sklearn.svm import SVC
import dash
import plotly.express as px
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from data_encoding import *
from pathlib import Path


def classifyModel(df1: pd.DataFrame) -> Dict:
    label_col = "FIRE_STATUS"
    feature_cols = df1.columns.tolist()
    feature_cols.remove(label_col)
    x = df1[feature_cols]
    y = df1[label_col]

    # train model
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # create and train SVM gaussian kernel
    svclassifier = SVC(kernel='rbf')
    svclassifier = svclassifier.fit(train_x, train_y)
    sv_model = svclassifier.predict(test_x)
    svm_confusion_matrix = confusion_matrix(test_y, sv_model)
    svm_accuracy = accuracy_score(test_y, sv_model)

    return dict(model=svclassifier, x=train_x, y=train_y, confusion_matrix=svm_confusion_matrix, accuracy=svm_accuracy)


def plot_learning_graph(record: Dict):
    # find learning curve of model
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(estimator=record['model'], X=record['x'], y=record['y'],
                                                            cv=cv, n_jobs=4)

    # Calculate training and test mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--',
             label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.title('Learning Curve')
    plt.xlabel('Training Data Size')
    plt.ylabel('Model accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()


def predictEvent(record: Dict, encode_state: DataFrame):

    df_svm = pd.DataFrame(record['confusion_matrix'], columns=['No', 'Yes'])
    svm_accuracy = record['accuracy']

    state_encode = encode_state

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # App layout
    app.layout = html.Div([
        html.H1("Wildfire Prediction Dashboard", style={'text-align': 'center'}),
        html.Hr(),
        html.Div(
            [
                dbc.Row([
                    dbc.Card(
                        [
                            dbc.CardHeader("Model Accuracy"),
                            dbc.CardBody(
                                html.P(id='card_id', className="card-text")
                            ),
                        ],
                        style={'text-align': 'center', 'margin-left': '8vw'}
                    )
                ]
                ),
                dbc.Row([dbc.Col(dcc.Graph(id='model-heatMap'))])
            ],
            style={
                'position': 'relative',
                'float': 'left',
                'width': '30%'
            }
        ),
        html.Div(
            [
                dbc.FormGroup([
                    dbc.Label("Choose State: "),
                    dcc.Dropdown(id="slct_state",
                                 options=[
                                     {"label": "California", "value": "CA"},
                                     {"label": "Colorado", "value": "CO"},
                                     {"label": "Nevada", "value": "NV"},
                                     {"label": "Texas", "value": "TX"}
                                 ],
                                 multi=False,
                                 value="CA",
                                 style={'margin-bottom': '1em'}
                                 ),
                ]),
                dbc.FormGroup([
                    html.Div(
                        [
                            "Temperature (K) : ",
                            dbc.Input(id='id_temp', type="number", value=273.16,
                                      placeholder="input values between 273.16 and 373.16")],
                        style={'margin-bottom': '1em'}
                    ),
                    html.Div(
                        [
                            "Humidity : ",
                            dbc.Input(id='id_humid', type="number", value=10, min=10, max=100, step=1)],
                        style={'margin-bottom': '1em'}
                    ),
                    html.Div(
                        [
                            "Pressure : ",
                            dbc.Input(id='id_pressure', type="number", value=100, min=100, max=10000, step=1)],
                        style={'margin-bottom': '1em'}
                    ),
                    html.Div(
                        [
                            "Wind Speed : ",
                            dbc.Input(id='id_wind', type="number", value=0.1, min=0, max=10, step=0.1)],
                        style={'margin-bottom': '1em'}
                    )
                ]),
                dbc.Button('Predict Fire', id='predict-button', color='primary', style=dict(width='90%',
                                                                                            verticalAlign="middle"),
                           block=True)
            ],
            style={
                'position': 'relative',
                'float': 'left',
                'width': '20%',
                'display': 'inline-block'
            }
        ),
        html.Div(
            [
                dbc.Row([
                    dbc.Col(dcc.Graph(id='predict-graph')),
                ])
            ],
            style={
                'position': 'absolute',
                'margin-left': '25px',
                'float': 'right',
                'width': '45%',
                'display': 'inline-block'
            }
        )
    ]
    )

    # Connect the Plotly graphs with Dash Components

    # populate the card with model's accuracy
    @app.callback(
        Output('card_id', 'children'),
        Input('predict-button', 'n_clicks')
    )
    def predict_graph(id):
        return str(round(svm_accuracy, 2) * 100) + '%'

    @app.callback(
        [Output(component_id='model-heatMap', component_property='figure'),
         Output(component_id='predict-graph', component_property='figure')],
        Input('predict-button', 'n_clicks'),
        [State('slct_state', 'value'),
         State('id_temp', 'value'),
         State('id_humid', 'value'),
         State('id_pressure', 'value'),
         State('id_wind', 'value')]
    )
    def predict_graph(n_clicks, state, temp, humid, pressure, wind):
        # Plotly confusion matrix in HeatMap
        col_corr = df_svm.corr().to_numpy()
        svm_arr = df_svm.values.tolist()
        figMap = ff.create_annotated_heatmap(col_corr, x=['No', 'Yes'], y=['No', 'Yes'],
                                             annotation_text=svm_arr, colorscale='RdBu')
        figMap.update_xaxes(side="top")

        # Plotly Fire Map
        try:
            state_data = {'state': state,
                          'temp': temp,
                          'humid': humid,
                          'pressure': pressure,
                          'wind': wind}
            dff = pd.DataFrame(state_data, index=[0])
            input_df = pd.DataFrame({'temp': temp, 'humid': humid, 'pressure': pressure, 'wind': wind}, index=[0])

            # normalize the input data
            norm_scale = preprocessing.MinMaxScaler().fit(input_df.transpose())
            df_std = norm_scale.transform(input_df.transpose())
            input_df = pd.DataFrame(np.transpose(df_std))
            input_df['code'] = state

            new_state = state_encode[state_encode['code'] == state]
            input_df = input_df.merge(new_state, on='code', how='left')
            input_df.pop('code')

            # predict ooutcome
            dff['predicted'] = record['model'].predict(input_df)

            dff['predicted_state'] = np.where(dff['predicted'] == 0, 'No', 'Yes')

            # Plotly Map
            fig = px.choropleth(
                data_frame=dff,
                locationmode='USA-states',
                locations='state',
                scope="usa",
                color='predicted',
                hover_data=['state', 'temp', 'humid', 'pressure', 'wind'],
                color_continuous_scale=px.colors.sequential.YlOrRd,
                labels={'predicted': 'Occurrence of Fire'},
                template='plotly_dark'
            )
        except Exception as error:
            print('Caught this error: ' + repr(error))

        return figMap, fig

    return app


def clean_user_data(df1: DataFrame) -> DataFrame:
    encode_state = df1.copy()
    encode_state.pop('temperature')
    encode_state.pop('humidity')
    encode_state.pop('pressure')
    encode_state.pop('wind_speed')
    encode_state.pop('FIRE_STATUS')
    state_list = encode_state.columns.tolist()
    new_list = []
    for name in state_list:
        name = name.replace('x0_', '')
        new_list.append(name)

    encode_state['code'] = 'na'
    for i in range(len(encode_state)):
        for j in new_list:
            if encode_state.iloc[i, new_list.index(j)] == 1:
                encode_state.iloc[i, len(new_list)] = j

    encode_state = encode_state.drop_duplicates()

    return encode_state


def run_process():
    abs_path = os.path.abspath('Data\\processed_data.csv')
    df = pd.read_csv(Path(abs_path))

    # create model
    model = classifyModel(df)
    state_clean = clean_user_data(df)

    # plot learning curve
    # plot_learning_graph(model)

    # show dashboard
    result = predictEvent(model, state_clean)

    return result


if __name__ == '__main__':
    my_app = run_process()
    my_app.run_server(debug=True)
