import pandas as pd

class Manual_Imputer:

    def __init__(self):
        self.manual_dict = {}

    def fit_transform(self, data, value='None', threshold=5):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            for i in data.columns:
                percent = ((data[i].isnull().sum()) / len(data[i])) * 100
                if percent <= threshold:
                    self.manual_dict[i] = data[i].mode().iloc[0]
                    data[i].fillna(data[i].mode().iloc[0], inplace=True)
                else:
                    self.manual_dict[i] = value
                    data[i].fillna(value, inplace=True)
        return data

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            for i in data.columns:
                data[i].fillna(self.manual_dict[i], inplace=True)
        return data
