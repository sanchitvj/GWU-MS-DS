import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Load time series data
df = pd.read_csv('weather_madrid_2019-2022 2.csv', parse_dates=["time"])
df = df.drop(['Unnamed: 0'], axis=1)
# print(df_orig.isnull().sum())
# print(df_orig.describe().to_string())
# print(df_orig.barometric_pressure.value_counts())

# df = df_orig.copy()
df["time"] = df["time"].dt.tz_localize(None)
df.index = df.time
# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    dcc.Graph(id="time-series-plot"),
    html.P('Select time range:'),
    dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed=df.index.min().date(),
        max_date_allowed=df.index.max().date(),
        start_date=df.index.min().date(),
        end_date=df.index.max().date(),
    )
])


# Define app callback
@app.callback(
    Output('ts-plot', 'figure'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_ts_plot(start_date, end_date):
    filtered_df = df.loc[start_date:end_date]
    fig = {
        'data': [{'x': filtered_df.index, 'y': filtered_df['value'], 'type': 'line'}],
        'layout': {'title': 'Time Series Plot'}
    }
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
