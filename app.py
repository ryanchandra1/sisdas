from dash import Dash, dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import pickle

# Load the training history
history_save_path = "history.pkl"
with open(history_save_path, 'rb') as file:
    history = pickle.load(file)

# Create the Dash app
app = Dash(__name__)

# Layout of the app
app.layout = html.Div(children=[
    html.H1(children='Model Training Results Dashboard'),

    dcc.Graph(
        id='accuracy-plot',
        figure=px.line(x=range(1, len(history) + 1), y=[x['val_acc'] for x in history],
                       labels={'x': 'Epoch', 'y': 'Accuracy'}, title='Validation Accuracy vs. Epochs')
    ),

    dcc.Graph(
        id='loss-plot',
        figure=px.line(x=range(1, len(history) + 1), y=[x['val_loss'] for x in history],
                       labels={'x': 'Epoch', 'y': 'Loss'}, title='Validation Loss vs. Epochs')
    ),

    html.Div(children=f'Test Accuracy: Not applicable'),

    # You may need to modify or replace the following line based on how you want to display confusion matrix
    # dcc.Graph(id='confusion-matrix', figure=...)
])


if __name__ == '__main__':
    app.run_server(debug=True)
