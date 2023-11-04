import pandas as pd


def get_info_function(df, col_name):
    return len(list(filter(lambda x: str(x) != "nan", pd.unique(df[col_name]))))
