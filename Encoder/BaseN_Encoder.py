import pandas as pd
from category_encoders.basen import BaseNEncoder


class BaseN_Encode:

    def __init__(self):
        self.baseN_encode = BaseNEncoder(base=10)

    def fit_transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            encode_data = self.baseN_encode.fit_transform(data)
        return encode_data

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            encode_data = self.baseN_encode.transform(data)
            return encode_data
