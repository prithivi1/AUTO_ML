import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Min_Max_Scaler:

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            scaled = self.scaler.fit_transform(data)
            dataframe = pd.DataFrame(scaled, columns=data.columns)
        return dataframe

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            scaled = self.scaler.transform(data)
            dataframe = pd.DataFrame(scaled, columns=data.columns)
        return dataframe
