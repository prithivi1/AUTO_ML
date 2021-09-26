import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class Label_Encode:

    def __init__(self):
        self.ordinal_encode = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

    def fit_transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            encode_data = self.ordinal_encode.fit_transform(data)
            data_imputed = pd.DataFrame(encode_data, columns=data.columns, index=data.index)
            data_imputed.fillna(50,inplace=True,axis=1)
        return data_imputed

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            encode_data = self.ordinal_encode.transform(data)
            data_imputed = pd.DataFrame(encode_data, columns=data.columns, index=data.index)
            data_imputed.fillna(50, inplace=True, axis=1)
            return data_imputed
