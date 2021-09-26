import pandas as pd
from category_encoders.target_encoder import TargetEncoder


class Target_Encode:

    def __init__(self):
        self.target_encode = TargetEncoder()

    def fit_transform(self, data, target_col):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            encode_data = self.target_encode.fit_transform(data, target_col)
            data_imputed = pd.DataFrame(encode_data, columns=data.columns, index=data.index)
            data_imputed.fillna(50, inplace=True, axis=1)
        return data_imputed

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            encode_data = self.target_encode.transform(data)
            data_imputed = pd.DataFrame(encode_data, columns=data.columns, index=data.index)
            return data_imputed
