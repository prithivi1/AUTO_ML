import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class One_Hot:

    def __init__(self):
        pass

    def fit_transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            onehot_encoder = OneHotEncoder(sparse=False,)
            encod_data = onehot_encoder.fit_transform(data)
            data_coded = pd.DataFrame(encod_data, columns=data.columns, index=data.index)
            return data_coded

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            one_hot_encoded_data = pd.get_dummies(data)
        return one_hot_encoded_data
