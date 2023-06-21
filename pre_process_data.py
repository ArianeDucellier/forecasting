import pandas as pd

def pre_process_data():
    df = pd.read_csv('Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv')
    df = df.drop(columns=['RegionID', 'SizeRank', 'RegionType', 'StateName'])
    df = df.rename(columns={'RegionName':'date'})
    df = df.set_index('date')
    df = df.stack().unstack(level=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.pct_change()
    df = df.iloc[1:]
    return df
