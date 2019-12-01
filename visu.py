#!/usr/bin/env python

import pickle
import argparse

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import numpy as np

import torch

from gan import Generator, reconstruct_sentence


def memoize(function):
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper


data = None
model = None
latent_size = 10
device = torch.device('cpu')


def load_data(filename):
    global data
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    data['reverse_mapping'] = {v: k for k, v in data['mapping'].items()}
    print('Loaded data')


def load_model(filename):
    checkpoint = torch.load(filename, map_location=device)
    global model
    model = Generator(
        len(data['mapping']),
        data['max_length'],
        latent_size)

    model.load_state_dict(checkpoint['generator_model_state_dict'])

    model.eval()
    print('Loaded model')


@memoize
def available_words():
    global data
    if data is None:
        return []
    words = [word for word in data['mapping']]
    words.sort()
    return words


def create_app():
    app = dash.Dash(__name__)
    app.title = 'Wiki Fake Visu'
    app.css.append_css({'external_url': '/assets/style.css'})

    app.layout = html.Div([
        html.Div([
            html.P('Test your sentence', className="control_label"),
            dcc.Input(id='input-1',
                      type='text',

                      value=''
                      ),
        ],
            className="pretty_container four columns"),
        html.Div([
            html.P('Create a fake sentence'),
            html.Button('Generate', id='generate'),
            dcc.Input(id='genetated', readOnly=True),
            dcc.Graph(id='prediction-heat')
        ], className="pretty_container four columns")
    ],
        style={
            "display": "flex",
            "flex-direction": "column"
    })

    @app.callback([Output('genetated', 'value'), Output(component_id='prediction-heat',  component_property='figure')],
                  [Input('generate', 'n_clicks')])
    def generate_sentence(n_clicks):
        global model
        if model is None:
            return 'Model not loaded yet'
        fake_l = torch.tensor([data['max_length']],
                              dtype=torch.long).to(device)
        z = torch.randn(1, latent_size).to(device)
        fake = model(z, fake_l)

        sentence = reconstruct_sentence(data['mapping'], fake.squeeze())

        fake = fake.detach().numpy()
        m = np.max(fake, axis=0)
        i = np.argsort(m)[-10:]
        fake = fake[:, i]
        fake = np.transpose(fake)

        figure = go.Heatmap(z=fake, y=[data['reverse_mapping'][j] for j in i])

        return sentence, {'data': [figure]}

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='wiki gan visu')
    parser.add_argument('-i', '--input-filename', type=str, required=True)
    parser.add_argument('-m', '--model-filename', type=str, required=True)
    args = parser.parse_args()
    load_data(args.input_filename)
    load_model(args.model_filename)
    app = create_app()
    app.run_server(debug=True)
