import numpy as np
import pandas as pd
import torch

from datetime import datetime
from dateutil.relativedelta import relativedelta
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from linear_regression import get_target, get_feature

def get_features(features, ds, n_months, sa):
    """
    Function to read the features time series.
    Input:
        features: list of strings. Names of the features. 
        ds: datetime. We only keep data anterior to ds.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        df: panda dataframe.
    """
    for count, feature in enumerate(features):
        feature_data = get_feature(feature, ds, n_months, sa)
        feature_data[feature] = feature_data[feature].shift(1)
        if count == 0:
            df = feature_data
        else:
            df = pd.merge(df, feature_data, on=['date'], how='inner')
        df = df.dropna()
    return df

def get_train_data(features, ds, n_months, sa):
    """
    Function to get the training set. We get the target and features
    and apply a lag to the features for the forecasting.
    Input:
        features: list of strings. Names of the features.
        ds: datetime. We only keep data anterior to ds.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        df: pandas dataframe.
    """
    target = get_target(ds, n_months, sa)
    df = get_features(features, ds, n_months, sa)
    df = pd.merge(target, df, on=['date'], how='inner')
    df = df.dropna()
    return df

def get_test_data(features, ds, n_months, sa):
    """
    Function to get the test set. We get the target and feature
    and apply a lag to the features for the forecasting.
    Input:
        features: list of strings. Names of the features.
        ds: datetime. We only keep data anterior to ds + 1 month.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        df: pandas dataframe.
    """
    ds_next = ds + relativedelta(months=1)
    target = get_target(ds_next, n_months, sa)
    df = get_features(features, ds_next, n_months, sa)
    df = pd.merge(target, df, on=['date'], how='inner')
    df = df.dropna()
    return df

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers):
        super().__init__()
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers
        )

        self.linear = nn.Linear(in_features=hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()

        return out

def train_step(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    model.train()
    
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def predict(data_loader, model):

    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)    
    return output

def train_model(df_train, hidden_units, num_layers, sequence_length, batch_size, learning_rate):
                
    train_dataset = SequenceDataset(
        df_train,
        target='target',
        features=list(df_train.columns.difference(['target', 'date'])),
        sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ShallowRegressionLSTM(num_features=len(features), \
        hidden_units=hidden_units, num_layers=num_layers)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for ix_epoch in range(n_epochs):
        train_step(train_loader, model, loss_function, optimizer=optimizer)

def test_model(df_test, model):

    test_dataset = SequenceDataset(
        df_test,
        target='target',
        features=list(df_test.columns.difference(['target', 'date'])),
        sequence_length=sequence_length
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    y_hat = predict(test_loader, model).numpy()

    return y_hat
