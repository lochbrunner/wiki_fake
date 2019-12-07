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

from gan import Generator, Discriminator, reconstruct_sentence

# This could be make session dependent via:
# https://dash.plot.ly/sharing-data-between-callbacks
current_sentence = ''


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
generator = None
discriminator = None
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
    global generator
    global discriminator
    generator = Generator(
        len(data['mapping']),
        data['max_length'],
        latent_size)

    generator.load_state_dict(checkpoint['generator_model_state_dict'])
    generator.eval()

    discriminator = Discriminator(len(data['mapping']))
    discriminator.load_state_dict(checkpoint['discriminator_model_state_dict'])
    discriminator.eval()

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
            html.Button('Test', id='test'),
            dcc.Input(id='test-sentence',
                      type='text',
                      value='',
                      ),
            html.P(id='validated-input', children=[]),
            html.P(id='validation-result')
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
        global generator
        if generator is None:
            return 'Model not loaded yet'
        fake_l = torch.tensor([data['max_length']],
                              dtype=torch.long).to(device)
        z = torch.randn(1, latent_size).to(device)
        fake = generator(z, fake_l)

        sentence = reconstruct_sentence(data['mapping'], fake.squeeze())

        fake = fake.detach().numpy()
        m = np.max(fake, axis=0)
        i = np.argsort(m)[-10:]
        fake = fake[:, i]
        fake = np.transpose(fake)

        figure = go.Heatmap(z=fake, y=[data['reverse_mapping'][j] for j in i])

        return sentence, {'data': [figure]}

    @app.callback(Output('validated-input', 'children'), [Input('test-sentence', 'value')])
    def validate_input(sentence):
        words = sentence.split()
        if len(words) == 0:
            return []
        items = [html.Span(word, className=(
            'in-valid' if word.lower() not in data['mapping'] else '')) for word in words]
        global current_sentence
        current_sentence = sentence
        return items

    @app.callback(Output('validation-result', 'children'),
                  [Input('test', 'n_clicks')])
    def check_is_real(n_clicks):
        global current_sentence
        mapping = data['mapping']
        words = current_sentence.split()
        for word in words:
            if word not in mapping:
                return f'Unknown word "{word}"'
        if len(words) == 0:
            return '-'

        x = torch.as_tensor([mapping[word]
                             for word in words], dtype=torch.long)
        pad_token = 0

        padded_x = np.ones((data['max_length']))*pad_token
        padded_x[0:len(x)] = x
        x = torch.as_tensor(padded_x, dtype=torch.long).to(device).unsqueeze(0)
        l = torch.as_tensor([len(words)], dtype=torch.long).to(
            device)
        print(f'l: {l.size()}')
        judge = discriminator(x, l).view(-1).item()
        return [html.Span(f'{judge*100:.3f}%')]

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
