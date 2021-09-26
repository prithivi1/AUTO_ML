import pandas as pd
import numpy as np

from Models.Classification import Classification_Helper_Script
from Models.Regression import Helper_Script


def select_continuous_data(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"The input param `{data}` must be of "
            "type pandas.DataFrame"
        )
    else:
        continuous_columns = []
        for i in data.columns:
            unique_percent = (len(data[i].unique()) / len(data[i])) * 100
            if unique_percent > 20:
                continuous_columns.append(i)
        return continuous_columns


def replace_other_nan_in_categorical_data(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"The input param `{data}` must be of "
            "type pandas.DataFrame"
        )
    else:
        data.replace(['None', 'none', '-', 'NONE'], np.nan, inplace=True)
    return data


def run_models(df, y, score, n_split, problem_type):
    if problem_type == 'Regression':
        models_df = Helper_Script.train_base_model(df, y, score, n_split)
    if problem_type == 'Classification':
        models_df = Classification_Helper_Script.train_base_model(df, y, score, n_split)
    return models_df
