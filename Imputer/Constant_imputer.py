import pandas as pd


class Constant_Imputer:

    def __init__(self):
        self.constant = None

    def fit_transform(self, data, value=0):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            self.contant = value
            for i in data.columns:
                data[i] = data[i].fillna(value)
            return data

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            for i in data.columns:
                data[i] = data[i].fillna(self.constant)
            return data
