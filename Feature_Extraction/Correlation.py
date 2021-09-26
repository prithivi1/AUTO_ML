import pandas as pd
import streamlit as st


class Correlation:

    def __init__(self):
        self.correlation_columns_to_drop = []

    def fit_transform(self, data, target_column, threshold=0.1):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            a = pd.concat([data, target_column], axis=1).corr()
            self.correlation_columns_to_drop = list(a[abs(a[target_column.name]) < threshold][target_column.name].index)

            if len(self.correlation_columns_to_drop) == 0:
                return data, abs(a[target_column.name]), self.correlation_columns_to_drop
            else:
                data.drop(self.correlation_columns_to_drop, axis=1, inplace=True)
                return data, abs(a[target_column.name]), self.correlation_columns_to_drop

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            if len(self.correlation_columns_to_drop) == 0:
                return data
            else:
                data.drop(self.correlation_columns_to_drop, axis=1, inplace=True)
                return data
