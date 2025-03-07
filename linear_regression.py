import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta
from math import ceil, sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score

def remove_seasonality(data):
    """
    Function to remove the seasonality from a 1D numpy array.
    Input:
        data: 1D numpy array. We assume we have monthly data and annual seasonality.
    Output:
        data_sa: 1D numpy array. Deseasonalized data.
        sa: 1D numpy array. Seasonality component.
    """
    data_extend = np.concatenate([np.array([data[0]] * 6), data, np.array([data[-1]] * 6)])
    weights = np.array([0.5] + [1] * 11 + [0.5]) / 12
    m = np.convolve(data_extend, weights, 'valid')
    data_demean = data - m
    N = int(ceil(len(data_demean) / 12)) * 12 - len(data_demean)
    data_demean = np.concatenate([data_demean, np.repeat(np.nan, N)])
    deviations = np.reshape(data_demean, (-1, 12)).transpose()
    deviations = np.nanmean(deviations, axis=1)
    deviations = deviations - np.mean(deviations)
    N = int(ceil(len(data_demean) / 12))
    s = np.tile(deviations, N)[0:len(data)]
    data_sa = data - s
    return (data_sa, s)

def get_target(ds, n_months, sa=False):
    """
    Function to read the home price time series and compute the growth.
    Input:
        ds: datetime. We only keep data anterior to ds.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        target_growth: pandas dataframe.
    """
    # Read data and reformat dataframe
    target = pd.read_csv('data/2025_02/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv')
    target = target.drop(columns=['RegionID', 'SizeRank', 'RegionType', 'StateName'])
    target = target.rename(columns={'RegionName':'index'})
    target = target.set_index('index')
    target = target.stack().unstack(level=0)
    target = target.reset_index()
    target = target.rename(columns={'index': 'date'})
    target['date'] = pd.to_datetime(target['date'])
    # Filter by date
    target = target.loc[target['date'] < ds]
    # Keep only the most recent data
    target = target.sort_values(['date'])
    target = target.iloc[-(n_months + 1):]
    # Compute growth
    date = target['date']
    target_growth = target.drop(columns=['date']).pct_change()
    target_growth['date'] = date
    target_growth = target_growth.iloc[1:]
    # Keep only national value
    target_growth = target_growth[['date', 'United States']]
    target_growth = target_growth.rename(columns={'United States':'target'})
    if sa == False:
        # Keep seasonal component
        return target_growth
    else:
        # Remove seasonality
        data = target_growth['target'].to_numpy()
        (data_sa, s) = remove_seasonality(data)
        target_growth['target'] = data_sa
        target_growth['seasonality'] = s
        return target_growth

def get_feature(name, ds, n_months, sa=False):
    """
    Function to read the feature time series.
    Input:
        name: string. Name of the feature.
        ds: datetime. We only keep data anterior to ds.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        feature: pandas dataframe.
    """
    # Read data and reformat dataframe
    feature = pd.read_csv('data/2025_02/' + name + '.csv')
    feature = feature.drop(columns=['RegionID', 'SizeRank', 'RegionType', 'StateName'])
    feature = feature.rename(columns={'RegionName':'index'})
    feature = feature.set_index('index')
    feature = feature.stack().unstack(level=0)
    feature = feature.reset_index()
    feature = feature.rename(columns={'index': 'date'})
    feature['date'] = pd.to_datetime(feature['date'])
    # Filter by date
    feature = feature.loc[feature['date'] < ds]
    # Sort
    feature = feature.sort_values(['date'])
    # Keep only the most recent data
    feature = feature.iloc[-n_months:]
    # Keep only national value
    feature = feature[['date', 'United States']]
    feature = feature.rename(columns={'United States':name})
    if sa == False:
        # Keep seasonal component
        return feature
    else:
        # Remove seasonality
        data = feature[name].to_numpy()
        (data_sa, s) = remove_seasonality(data)
        feature[name] = data_sa
        return feature

def get_train_data(name, lag, ds, n_months, sa=False):
    """
    Function to get the training set. We get the target and feature
    and apply a lag to the feature for the forecasting.
    Input:
        name: string. Name of the feature.
        lag: integer. Lag (in months) to apply to the feature.
        ds: datetime. We only keep data anterior to ds.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component from target and feature.
    Output:
        df: pandas dataframe.
    """
    target = get_target(ds, n_months, sa)
    feature = get_feature(name, ds, n_months, sa)
    feature[name] = feature[name].shift(lag)
    df = pd.merge(target, feature, on=['date'], how='inner')
    df = df.dropna()
    return df

def train_model(name, lag, ds, n_months, sa=False):
    """
    Function to train the model.
    Input:
        name: string. Name of the feature.
        lag: integer. Lag (in months) to apply to the feature.
        ds: datetime. We only keep data anterior to ds.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        model: scikit-learn model for linear regression.
    """
    data = get_train_data(name, lag, ds, n_months, sa)
    X = data[name].to_numpy().reshape(-1, 1)
    y = data['target'].to_numpy()
    model = LinearRegression()
    model.fit(X, y)
    return model

def get_test_data(name, lag, ds, n_months, sa=False):
    """
    Function to get the test set. We get the target and feature
    and apply a lag to the feature for the forecasting.
    Input:
        name: string. Name of the feature.
        lag: integer. Lag (in months) to apply to the feature.
        ds: datetime. We only keep data equal to ds + 1 month.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        df: pandas dataframe.
    """
    ds_next = ds + relativedelta(months=1)
    target = get_target(ds_next, n_months, sa)
    feature = get_feature(name, ds_next, n_months, sa)
    feature[name] = feature[name].shift(lag)
    df = pd.merge(target, feature, on=['date'], how='inner')
    df = df.dropna()
    df = df.loc[df['date'] >= ds]
    return df

def test_model(model, name, lag, ds, n_months, sa=False):
    """
    Function to test the model.
    Input:
        model: scikit-learn model for linear regression.
        name: string. Name of the feature.
        lag: integer. Lag (in months) to apply to the feature.
        ds: datetime. We only keep equal to ds + 1 month.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        y: 1D numpy array. Actual values.
        y_hat: 1D numpy array. Predicted values.
    """
    data = get_test_data(name, lag, ds, n_months, sa)
    X = data[name].to_numpy().reshape(-1, 1)
    y = data['target'].to_numpy()
    y_hat = model.predict(X)
    if sa == True:
        y = y + data['seasonality'].to_numpy()
        y_hat = y_hat + data['seasonality'].to_numpy()
    return (y, y_hat)

def backtest(name, lag, ds_begin, ds_end, n_months, sa=False):
    """
    Function to backtest the model.
    We train the data for each month between ds_begin and ds_end
    and compute the 1-month-ahead prediction for each model.
    Input:
        name: string. Name of the feature.
        lag: integer. Lag (in months) to apply to the feature.
        ds_begin: datetime. First date for which we test the model.
        ds_end: datetime. Last date for which we test the model.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        y: 1D numpy array. Actual values.
        y_hat: 1D numpy array. Predicted values.
    """
    ds_range = pd.date_range(start=ds_begin, end=ds_end, freq='MS')
    y = np.zeros(len(ds_range))
    y_hat = np.zeros(len(ds_range))
    for i, ds in enumerate(ds_range):
        model = train_model(name, lag, ds, n_months, sa)
        result = test_model(model, name, lag, ds, n_months, sa)
        y[i] = result[0][0]
        y_hat[i] = result[1][0]
    return (y, y_hat)

if __name__ == '__main__':

    features = ['Metro_sales_count_now_uc_sfrcondo_month',
                'Metro_invt_fs_uc_sfrcondo_month',
                'Metro_new_listings_uc_sfrcondo_month',
                'Metro_mean_doz_pending_uc_sfrcondo_month',
                'Metro_mean_sale_to_list_uc_sfrcondo_month',
                'Metro_med_doz_pending_uc_sfrcondo_month',
                'Metro_median_sale_to_list_uc_sfrcondo_month',
                'Metro_new_pending_uc_sfrcondo_month',
                'Metro_perc_listings_price_cut_uc_sfrcondo_month',
                'Metro_pct_sold_above_list_uc_sfrcondo_month',
                'Metro_pct_sold_below_list_uc_sfrcondo_month']
    ds_begin = datetime(2022, 7, 1)
    ds_end = datetime(2024, 12, 1)
    n_months = 48
    for feature in features:
        # Predictions when removing the seasonality
        MSE = np.zeros(6)
        RMSE = np.zeros(6)
        MAE = np.zeros(6)
        MAPE = np.zeros(6)
        R2 = np.zeros(6)
        for lag in range(1, 7):
            print('SA', feature, lag)
            (y, y_hat) = backtest(feature, lag, ds_begin, ds_end, n_months, True)
            MSE[lag - 1] = mean_squared_error(y, y_hat)
            RMSE[lag - 1] = sqrt(mean_squared_error(y, y_hat))
            MAE[lag - 1] = mean_absolute_error(y, y_hat)
            MAPE[lag - 1] = mean_absolute_percentage_error(y, y_hat)
            R2[lag - 1] = r2_score(y, y_hat)
        error = pd.DataFrame({'lag': np.arange(-1, -7, -1), \
            'MSE': MSE, 'RMSE': RMSE, 'MAE': MAE, 'MAPE': MAPE, 'R2': R2})
        error.to_csv('linear_regression/' + feature + '_sa.csv')
        # Predictions when keeping the seasonality
        MSE = np.zeros(6)
        RMSE = np.zeros(6)
        MAE = np.zeros(6)
        MAPE = np.zeros(6)
        R2 = np.zeros(6)
        for lag in range(1, 7):
            print('NSA', feature, lag)
            (y, y_hat) = backtest(feature, lag, ds_begin, ds_end, n_months, False)
            MSE[lag - 1] = mean_squared_error(y, y_hat)
            RMSE[lag - 1] = sqrt(mean_squared_error(y, y_hat))
            MAE[lag - 1] = mean_absolute_error(y, y_hat)
            MAPE[lag - 1] = mean_absolute_percentage_error(y, y_hat)
            R2[lag - 1] = r2_score(y, y_hat)
        error = pd.DataFrame({'lag': np.arange(-1, -7, -1), \
            'MSE': MSE, 'RMSE': RMSE, 'MAE': MAE, 'MAPE': MAPE, 'R2': R2})
        error.to_csv('linear_regression/' + feature + '.csv')
 