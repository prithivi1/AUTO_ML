import pandas as pd


class Numerical_Categorical_Split:
    def __init__(self):
        self.numerical_columns = []
        self.categorical_columns = []

    def split_numerical_categorical_fit_transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            self.numerical_columns = data.select_dtypes(['int', 'float']).columns
            self.categorical_columns = data.select_dtypes(['object']).columns

            numerical_data = data[self.numerical_columns]
            categorical_data = data[self.categorical_columns]

        return numerical_data, categorical_data

    def split_numerical_categorical_transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            numerical_data = data[self.numerical_columns]
            categorical_data = data[self.categorical_columns]

        return numerical_data, categorical_data
