from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd

class Mice_Imputer:

    def __init__(self):
        pass

    def fit_transform(self, data, model=GradientBoostingRegressor()):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            self.iter_imp_numeric = IterativeImputer(model)
            imputed_train = self.iter_imp_numeric.fit_transform(data)
            data_imputed = pd.DataFrame(imputed_train, columns=data.columns, index=data.index)
        return data_imputed

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            imputed_train = self.iter_imp_numeric.transform(data)
            data_imputed = pd.DataFrame(imputed_train, columns=data.columns, index=data.index)
        return data_imputed
