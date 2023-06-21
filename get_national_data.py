import pandas as pd

from functools import reduce

def get_national_data(filename, varname, growth=False):
    df = pd.read_csv(filename)
    df = df.drop(columns=['RegionID', 'SizeRank', 'RegionType', 'StateName'])
    df = df.rename(columns={'RegionName':'index'})
    df = df.set_index('index')
    df = df.stack().unstack(level=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if growth:
        df = df.pct_change()
        df = df.iloc[1:]
    df = df.reset_index()
    df = df.rename(columns={'index': 'date', 'United States': varname})
    df = df[['date', varname]]
    return df

def merge_data():
    zhvi = get_national_data('Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv', 'zhvi', True)
    sales = get_national_data('Metro_sales_count_now_uc_sfrcondo_month.csv', 'sales', False)
    invt = get_national_data('Metro_invt_fs_uc_sfrcondo_month.csv', 'invt', False)
    df = reduce(lambda x, y: x.merge(y, on='date'), [zhvi, sales, invt])
    df['ratio'] = df['sales'] / df['invt']
    return df
