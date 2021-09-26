import pandas as pd


class Frequency_Encoding:

    def __init__(self):
        self.encoder_dictionary = {}

    def fit_transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            for i in data.columns:
                self.encoder_dictionary[i] = data[i].value_counts().to_dict()
            for i in data.columns:
                data[i] = data[i].map(self.encoder_dictionary[i])
        return data

    def transform(self, data, unknown_data=0):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            for i in data.columns:
                data[i] = data[i].map(self.encoder_dictionary[i])
                data[i].fillna(unknown_data, inplace=True)

        return data
