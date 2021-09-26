import pandas as pd

class Manual_Numeric:

    def __init__(self):
        self.manual_dict = {}

    def fit_transform(self, data, threshold=5):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{self}` must be of "
                "type pandas.DataFrame"
            )
        else:
            for i in data.columns:
                percent = ((data[i].isnull().sum()) / len(data[i])) * 100

                if percent <= threshold:
                    self.manual_dict[i] = data[i].mean()
                    data[i].fillna(data[i].mean(), inplace=True)
                else:
                    self.manual_dict[i] = 0
                    data[i].fillna(0, inplace=True)
        return data

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{self}` must be of "
                "type pandas.DataFrame"
            )
        else:
            for i in data.columns:
                data[i].fillna(self.manual_dict[i], inplace=True)
        return data
