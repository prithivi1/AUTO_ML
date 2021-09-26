from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np


class Simple_Imputer:

    def __init__(self):
        pass

    def fit_transform(self, data, strategy):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            self.simple_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
            self.simple_imputer.fit(data)
            imputed_df = self.simple_imputer.transform(data)
            dataframe = pd.DataFrame(imputed_df, columns=data.columns)
        return dataframe

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            imputed_df = self.simple_imputer.transform(data)
            dataframe = pd.DataFrame(imputed_df, columns=data.columns)
        return dataframe
