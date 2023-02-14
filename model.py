import torch
import torch.nn.functional as F

from torch import nn


class LinearLayer(nn.Module):

    def __init__(self, in_size, out_size, dropout):
        super(LinearLayer, self).__init__()

        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        o = self.linear(x)
        o = self.relu(o)
        o = self.dropout(o)
        return o


class SegmentationModel(nn.Module):

    def __init__(self, params):

        super(SegmentationModel, self).__init__()

        self.params = params

        # Recurrent network
        self.lstm = nn.LSTM(input_size=params['input_size'],
                            hidden_size=params['hidden_size'],
                            num_layers=params['num_lstm'],
                            batch_first=params['batch_first'],
                            dropout=params['dropout'],
                            bidirectional=params['bidirectional'])

        # Linear Layers
        # This is a dynamic algorithm for creating linear networks.
        linear_layers = []
        in_size_linear = params['hidden_size']

        if params['bidirectional']:
            in_size_linear = in_size_linear * 2

        for i in range(1, params['num_linear'] + 1):

            out_size_linear = int(in_size_linear / (2 * i))

            if out_size_linear <= params['num_classes']:
                out_size_linear = params['num_classes']
                linear_layers.append(nn.Linear(in_size_linear, out_size_linear))
                break

            linear_layers.append(LinearLayer(in_size_linear,
                                             out_size_linear,
                                             params['dropout']))

            in_size_linear = out_size_linear

        # Add output layer if needed
        if out_size_linear != params['num_classes']:
            linear_layers.append(nn.Linear(out_size_linear, params['num_classes']))

        self.linear = nn.Sequential(*linear_layers)

    def compute_predictions(self, logits):
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)
        preds = probs.argmax(dim=-1)

        return preds

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        # TO IGNORE PADDING USE ignore_index = 100
        cross_entropy_loss = F.cross_entropy(logits.view(-1, self.params['num_classes']),
                                             labels.view(-1),
                                             ignore_index=0)
        return cross_entropy_loss

    def predict(self, x, device='cpu'):
        """
        Same as a forward pass, but just returns prediction on input samples. If
        you need to compute loss or logits, use the forward method.
        """

        inputs = x.to(device)
        rnn_out = self.lstm(inputs)[0]
        logits = self.linear(rnn_out[:, -1, :])
        preds = self.compute_predictions(logits)

        return preds

    def forward(self, batch, device='cpu', compute_preds=False, compute_loss=True):

        inputs, labels = batch['embeds'].to(device), batch['labels'].to(device)
        lstm_out = self.lstm(inputs)[0]

        logits = self.linear(lstm_out)
        out = {'logits': logits}

        if compute_preds:
            out['preds'] = self.compute_predictions(logits)

        if compute_loss:
            out['loss'] = self.compute_loss(logits, labels)

        return out
