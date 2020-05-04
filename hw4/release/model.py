import torch
from torch import nn
class GRUSentiment(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(GRUSentiment, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(embedding_dim,
                           hidden_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout = 0 if num_layers < 2 else dropout)
        # self.classifier = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, inputs):
        self.gru.flatten_parameters()
        inputs = self.embedding(inputs)
        x, _ = self.gru(inputs, None)
        # x dimension (batch, seq_len, hidden_size)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x

class LSTM_Net(nn.Module):
    def __init__(self, embedding_dim, num_layers):
        super(LSTM_Net, self).__init__()
        self.num_layers = num_layers
        # self.classifier = nn.Sequential( nn.Linear(embedding_dim, 512),
                                         # nn.Linear(512, 128),
                                         # nn.Linear(128, 1),
                                         # nn.Sigmoid() )
        self.classifier = nn.Sequential( nn.Linear(embedding_dim, 512),
                                         nn.Linear(512, 128),
                                         nn.Linear(128, 1) )
    def forward(self, inputs):
        x = self.classifier(inputs.float())
        return x

class LSTM(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, 1) )
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x
