import pandas as pd
from sklearn.preprocessing import StandardScaler


class Standard_Scaler:

    def __init__(self):
        self.stand = StandardScaler()

    def fit_transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            temp = self.stand.fit_transform(data)
            dataframe = pd.DataFrame(temp, columns=data.columns)
        return dataframe

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            temp = self.stand.transform(data)
            dataframe = pd.DataFrame(temp, columns=data.columns)
        return dataframe

