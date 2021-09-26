import pandas as pd


class Mode_Imputer:

    def __init__(self):
        self.mode_dict = {}

    def fit_transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            for i in data.columns:
                self.mode_dict[i] = data[i].mode().iloc[0]
                data[i].fillna(data[i].mode().iloc[0], inplace=True)
        return data

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            for i in data.columns:
                data[i].fillna(self.mode_dict[i], inplace=True)
            return data
