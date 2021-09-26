import pandas as pd
from category_encoders.hashing import HashingEncoder


class Hash_Encode:

    def __init__(self):
        self.hash_encode = HashingEncoder()

    def fit_transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            encode_data = self.hash_encode.fit_transform(data)
            # data_imputed = pd.DataFrame(encode_data, columns=data.columns, index=data.index)
        return encode_data

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            encode_data = self.hash_encode.transform(data)
            # data_imputed = pd.DataFrame(encode_data, columns=data.columns, index=data.index)
            return encode_data
