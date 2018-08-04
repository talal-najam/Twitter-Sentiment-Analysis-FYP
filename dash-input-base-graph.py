import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Sentiment Graph'),

    html.Div(children='''
        Dash: A web application framework
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1,2,3], 'y': [4,1,2], 'type': 'line', 'name': 'sf'},
                {'x': [1,2,3], 'y': [2,4,5], 'type': 'bar', 'name': 'karachi'},
            ],
            'layout': {
                'title': 'Sentiment Visualization Graph'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)