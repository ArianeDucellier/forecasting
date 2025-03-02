import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta
from math import sqrt
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

from linear_regression import get_target, get_feature

def get_features(features, ds, n_months, sa):
    """
    Function to read the features time series.
    Input:
        features: list of tuples (name of the feature, lag).
        ds: datetime. We only keep data anterior to ds.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        df: panda dataframe.
    """
    for count, feature in enumerate(features):
        name = feature[0]
        lag = feature[1]
        feature_data = get_feature(name, ds, n_months, sa)
        feature_data[name] = feature_data[name].shift(lag)
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
        features: list of tuples (name of the feature, lag).
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

def train_model(features, n_estimators, max_samples, ds, n_months, sa=False):
    """
    Function to train the model
    Input:
        name: string. Name of the feature.
        lag: integer. Lag (in months) to apply to the feature.
        ds: datetime. We only keep data anterior to ds.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        model: scikit-learn model for linear regression.
        scaler: scikit-learn model for normalizing features.
    """
    data = get_train_data(features, ds, n_months, sa)
    if sa == False:
        X = data.drop(columns=['date', 'target']).to_numpy()
    else:
        X = data.drop(columns=['date', 'target', 'seasonality']).to_numpy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    y = data['target'].to_numpy()
    model = BaggingRegressor(estimator=LinearRegression(),
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=len(features),
        oob_score=True,
        random_state=0)
    model.fit(X_scaled, y)
    return (model, scaler)

def get_test_data(features, ds, n_months, sa=False):
    """
    Function to get the test set. We get the target and feature
    and apply a lag to the features for the forecasting.
    Input:
        features: list of tuples (name of the feature, lag).
        ds: datetime. We only keep data equal to ds + 1 month.
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
    df = df.loc[df['date'] >= ds]
    return df

def test_model(model, scaler, features, ds, n_months, sa=False):
    """
    Function to test the model.
    Input:
        model: scikit-learn model for linear regression.
        features: list of tuples (name of the feature, lag).
        ds: datetime. We only keep equal to ds + 1 month.
        n_months: integer. Number of months to keep in the dataset.
        sa: boolean. If True, remove seasonal component.
    Output:
        y: 1D numpy array. Actual values.
        y_hat: 1D numpy array. Predicted values.
    """
    data = get_test_data(features, ds, n_months, sa)
    if sa == False:
        X = data.drop(columns=['date', 'target']).to_numpy()
    else:
        X = data.drop(columns=['date', 'target', 'seasonality']).to_numpy()
    X_scaled = scaler.transform(X)
    y = data['target'].to_numpy()
    y_hat = model.predict(X_scaled)
    if sa == True:
        y = y + data['seasonality'].to_numpy()
        y_hat = y_hat + data['seasonality'].to_numpy()
    return (y, y_hat)

def backtest(features, n_estimators, max_samples, ds_begin, ds_end, n_months, sa=False):
    """
    Function to backtest the model.
    We train the data for each month between ds_begin and ds_end
    and compute the 1-month-ahead prediction for each model.
    Input:
        features: list of tuples (name of the feature, lag).
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
        (model, scaler) = train_model(features, n_estimators, max_samples, ds, n_months, sa)
        result = test_model(model, scaler, features, ds, n_months, sa)
        y[i] = result[0][0]
        y_hat[i] = result[1][0]
    return (y, y_hat)

if __name__ == '__main__':

    ds_begin = datetime(2022, 7, 1)
    ds_end = datetime(2024, 12, 1)
    n_months = 48
    n_estimators = [10, 20, 50, 100, 200]
    max_samples = [0.5, 0.6, 0.7, 0.8, 0.9]
    # Predictions when removing the seasonality
    features_sa = [('Metro_sales_count_now_uc_sfrcondo_month', 1),
                   ('Metro_invt_fs_uc_sfrcondo_month', 1),
                   ('Metro_new_listings_uc_sfrcondo_month', 1),
                   ('Metro_mean_doz_pending_uc_sfrcondo_month', 1),
                   ('Metro_mean_sale_to_list_uc_sfrcondo_month', 2),
                   ('Metro_med_doz_pending_uc_sfrcondo_month', 1),
                   ('Metro_median_sale_to_list_uc_sfrcondo_month', 2),
                   ('Metro_new_pending_uc_sfrcondo_month', 1),
                   ('Metro_perc_listings_price_cut_uc_sfrcondo_month', 1),
                   ('Metro_pct_sold_above_list_uc_sfrcondo_month', 2),
                   ('Metro_pct_sold_below_list_uc_sfrcondo_month', 2)]
    MSE = np.zeros((5, 5))
    RMSE = np.zeros((5, 5))
    MAE = np.zeros((5, 5))
    MAPE = np.zeros((5, 5))
    R2 = np.zeros((5, 5))
    errors = []
    for i, n_estimator in enumerate(n_estimators):
        for j, max_sample in enumerate(max_samples):
            print('SA', n_estimator, max_sample)
            (y, y_hat) = backtest(features_sa, n_estimator, max_sample, \
                                  ds_begin, ds_end, n_months, True)
            MSE[i, j] = mean_squared_error(y, y_hat)
            RMSE[i, j] = sqrt(mean_squared_error(y, y_hat))
            MAE[i, j] = mean_absolute_error(y, y_hat)
            MAPE[i, j] = mean_absolute_percentage_error(y, y_hat)
            R2[i, j] = r2_score(y, y_hat)
            error = pd.DataFrame({'n_estimators': [n_estimator], \
                'max_samples': [max_sample], 'MSE': [MSE[i, j]], \
                'RMSE': [RMSE[i, j]], 'MAE': [MAE[i, j]], \
                'MAPE': [MAPE[i, j]], 'R2': [R2[i, j]]})
            errors.append(error)
    errors = pd.concat(errors)
    errors.to_csv('ensemble_LR/errors_sa.csv')
    # Predictions when keeping the seasonality
    features_nsa = [('Metro_sales_count_now_uc_sfrcondo_month', 1),
                   ('Metro_invt_fs_uc_sfrcondo_month', 1),
                   ('Metro_new_listings_uc_sfrcondo_month', 1),
                   ('Metro_mean_doz_pending_uc_sfrcondo_month', 1),
                   ('Metro_mean_sale_to_list_uc_sfrcondo_month', 2),
                   ('Metro_med_doz_pending_uc_sfrcondo_month', 1),
                   ('Metro_median_sale_to_list_uc_sfrcondo_month', 2),
                   ('Metro_new_pending_uc_sfrcondo_month', 1),
                   ('Metro_perc_listings_price_cut_uc_sfrcondo_month', 1),
                   ('Metro_pct_sold_above_list_uc_sfrcondo_month', 2),
                   ('Metro_pct_sold_below_list_uc_sfrcondo_month', 2)]
    MSE = np.zeros((5, 5))
    RMSE = np.zeros((5, 5))
    MAE = np.zeros((5, 5))
    MAPE = np.zeros((5, 5))
    R2 = np.zeros((5, 5))
    errors = []
    for i, n_estimator in enumerate(n_estimators):
        for j, max_sample in enumerate(max_samples):
            print('NSA', n_estimator, max_sample)
            (y, y_hat) = backtest(features_nsa, n_estimator, max_sample, \
                                  ds_begin, ds_end, n_months, False)
            MSE[i, j] = mean_squared_error(y, y_hat)
            RMSE[i, j] = sqrt(mean_squared_error(y, y_hat))
            MAE[i, j] = mean_absolute_error(y, y_hat)
            MAPE[i, j] = mean_absolute_percentage_error(y, y_hat)
            R2[i, j] = r2_score(y, y_hat)
            error = pd.DataFrame({'n_estimators': [n_estimator], \
                'max_samples': [max_sample], 'MSE': [MSE[i, j]], \
                'RMSE': [RMSE[i, j]], 'MAE': [MAE[i, j]], \
                'MAPE': [MAPE[i, j]], 'R2': [R2[i, j]]})
            errors.append(error)
    errors = pd.concat(errors)
    errors.to_csv('ensemble_LR/errors_nsa.csv')
