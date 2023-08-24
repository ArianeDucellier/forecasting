import numpy as np
import pandas as pd
import torch

from datetime import datetime
from dateutil.relativedelta import relativedelta
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score
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
    target = get_target(ds_next, n_months + 1, sa)
    df = get_features(features, ds_next, n_months + 1, sa)
    df = pd.merge(target, df, on=['date'], how='inner')
    df = df.dropna()
    return df

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.target = target
        self.features = features
        self.sequence_length = sequence_length
        self.X = torch.tensor(dataframe[features].values).float()
        self.y = torch.tensor(dataframe[target].values).float()

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
    total_loss = 0
    model.train()    
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss

def test_step(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    return avg_loss

def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)    
    return output

def train_model(df_train, df_test, hidden_units, num_layers, sequence_length, batch_size, learning_rate, n_epochs, sa):

    target_mean = df_train['target'].mean()
    target_stdev = df_train['target'].std()

    if sa == False:
        columns = list(df_train.columns.difference(['date']))
    else:
        columns = list(df_train.columns.difference(['date', 'seasonality']))
    for c in columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()
        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev

    if sa == False:
        features=list(df_train.columns.difference(['target', 'date']))
    else:
        features=list(df_train.columns.difference(['target', 'date', 'seasonality']))

    train_dataset = SequenceDataset(
        df_train,
        target='target',
        features=features,
        sequence_length=sequence_length
    )

    test_dataset = SequenceDataset(
        df_test,
        target='target',
        features=features,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ShallowRegressionLSTM(num_features=len(features), \
        hidden_units=hidden_units, num_layers=num_layers)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    avg_loss = np.zeros((n_epochs, 2))

    for ix_epoch in range(n_epochs):
        avg_train_loss = train_step(train_loader, model, loss_function, optimizer=optimizer)
        avg_test_loss = test_step(test_loader, model, loss_function)
        avg_loss[ix_epoch, 0] = avg_train_loss
        avg_loss[ix_epoch, 1] = avg_test_loss

    return (model, df_train, df_test, target_mean, target_stdev, avg_loss)

def test_model(df_train, df_test, model, sequence_length, batch_size, target_mean, target_stdev, sa):

    if sa == False:
        features=list(df_train.columns.difference(['target', 'date']))
    else:
        features=list(df_train.columns.difference(['target', 'date', 'seasonality']))

    train_dataset = SequenceDataset(
        df_train,
        target='target',
        features=features,
        sequence_length=sequence_length
    )

    test_dataset = SequenceDataset(
        df_test,
        target='target',
        features=features,
        sequence_length=sequence_length
    )

    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    df_train['target_predict'] = predict(train_eval_loader, model).numpy()
    df_test['target_predict'] = predict(test_loader, model).numpy()

    df_train['target_predict'] = df_train['target_predict'] * target_stdev + target_mean
    df_test['target_predict'] = df_test['target_predict'] * target_stdev + target_mean

    df_train['target'] = df_train['target'] * target_stdev + target_mean
    df_test['target'] = df_test['target'] * target_stdev + target_mean

    if sa == True:
        df_train['target'] = df_train['target'] + df_train['seasonality']
        df_test['target'] = df_test['target'] + df_test['seasonality']

        df_train['target_predict'] = df_train['target_predict'] + df_train['seasonality']
        df_test['target_predict'] = df_test['target_predict'] + df_test['seasonality']
        
    return (df_train, df_test)

def backtest(features, hidden_units, num_layers, sequence_length, batch_size, learning_rate, n_epochs, ds_begin, ds_end, n_months, sa=False):
    
    ds_range = pd.date_range(start=ds_begin, end=ds_end, freq='MS')
    y = np.zeros(len(ds_range))
    y_hat = np.zeros(len(ds_range))
    for i, ds in enumerate(ds_range):
        print(ds)
        df_train = get_train_data(features, ds, n_months, sa)
        df_test = get_test_data(features, ds, n_months, sa)
        (model, df_train, df_test, target_mean, target_stdev, avg_loss) = train_model(df_train, df_test, hidden_units, num_layers, sequence_length, batch_size, learning_rate, n_epochs, sa)
        (df_train, df_test) = test_model(df_train, df_test, model, sequence_length, batch_size, target_mean, target_stdev, sa)
        y[i] = df_test['target'].iloc[-1]
        y_hat[i] = df_test['target_predict'].iloc[-1]
    return (y, y_hat)

if __name__ == '__main__':

    ds_begin = datetime(2021, 7, 1)
    ds_end = datetime(2023, 5, 1)
    n_months = 36
    hidden_units = [10, 12, 14]
    num_layers = [1, 2, 3]
    sequence_lengths = [12, 18, 24]
    batch_size = 4
    learning_rate = 5e-4
    n_epochs = 100
    features = ['sales_raw',
                'inventory_raw',
                'new_listings_raw',
                'mean_days_to_pending_raw',
                'mean_sale_to_list_ratio_raw',
                'median_days_to_pending_raw',
                'median_sale_to_list_ratio_raw',
                'newly_pending_listings_raw',
                'pct_listings_price_cut_raw',
                'pct_sold_above_list_price_raw',
                'pct_sold_below_list_price_raw']
    MSE = np.zeros((3, 3, 3))
    RMSE = np.zeros((3, 3, 3))
    MAE = np.zeros((3, 3, 3))
    MAPE = np.zeros((3, 3, 3))
    R2 = np.zeros((3, 3, 3))
    errors = []
    # Predictions when removing the seasonality
    for i, hidden_unit in enumerate(hidden_units):
        for j, num_layer in enumerate(num_layers):
            for k, sequence_length in enumerate(sequence_lengths):
                print('SA', hidden_unit, num_layer, sequence_length)
                (y, y_hat) = backtest(features, hidden_unit, num_layer, sequence_length, batch_size, learning_rate, n_epochs, ds_begin, ds_end, n_months, sa=True)
                MSE[i, j, k] = mean_squared_error(y, y_hat)
                RMSE[i, j, k] = sqrt(mean_squared_error(y, y_hat))
                MAE[i, j, k] = mean_absolute_error(y, y_hat)
                MAPE[i, j, k] = mean_absolute_percentage_error(y, y_hat)
                R2[i, j, k] = r2_score(y, y_hat)
                error = pd.DataFrame({'hidden_units': [hidden_unit], \
                    'num_layers': [num_layer], 'sequence_lengths': [sequence_length], \
                    'MSE': [MSE[i, j, k]], 'RMSE': [RMSE[i, j, k]], \
                    'MAE': [MAE[i, j, k]], 'MAPE': [MAPE[i, j, k]], \
                    'R2': [R2[i, j, k]]})
                errors.append(error)
    errors = pd.concat(errors)
    errors.to_csv('LSTM/errors_sa.csv')
    # Predictions when keeping the seasonality
#    for i, hidden_unit in enumerate(hidden_units):
#        for j, num_layer in enumerate(num_layers):
#            for k, sequence_length in enumerate(sequence_lengths):
#                print('NSA', hidden_unit, num_layer, sequence_length)
#                (y, y_hat) = backtest(features, hidden_unit, num_layer, sequence_length, batch_size, learning_rate, n_epochs, ds_begin, ds_end, n_months, sa=False)
#                MSE[i, j, k] = mean_squared_error(y, y_hat)
#                RMSE[i, j, k] = sqrt(mean_squared_error(y, y_hat))
#                MAE[i, j, k] = mean_absolute_error(y, y_hat)
#                MAPE[i, j, k] = mean_absolute_percentage_error(y, y_hat)
#                R2[i, j, k] = r2_score(y, y_hat)
#                error = pd.DataFrame({'hidden_units': [hidden_unit], \
#                    'num_layers': [num_layer], 'sequence_lengths': [sequence_length], \
#                    'MSE': [MSE[i, j, k]], 'RMSE': [RMSE[i, j, k]], \
#                    'MAE': [MAE[i, j, k]], 'MAPE': [MAPE[i, j, k]], \
#                    'R2': [R2[i, j, k]]})
#                errors.append(error)
#    errors = pd.concat(errors)
#    errors.to_csv('LSTM/errors_nsa.csv')
