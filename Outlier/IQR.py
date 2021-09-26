import pandas as pd
from Utils.Utility_Functions import select_continuous_data


class IQR_Treatement:

    def __init__(self):
        self.continuous_columns = []
        self.lower_range = {}
        self.upper_range = {}

    def fit_transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            self.continuous_columns = select_continuous_data(data)

            for i in self.continuous_columns:
                q1 = data[i].quantile(0.25)
                q3 = data[i].quantile(0.75)

                IQR = q3 - q1

                lower_range = q1 - 1.5 * IQR
                upper_range = q3 + 1.5 * IQR

                print(" {:20s} {:5s}".format(data[i].name,
                                             str((data[i] < lower_range).sum() + (data[i] > upper_range).sum())))

                train_low_index = data[i][data[i] < lower_range].index
                train_high_index = data[i][data[i] > upper_range].index

                data[i][train_low_index] = lower_range
                data[i][train_high_index] = upper_range

                self.lower_range[i] = lower_range
                self.upper_range[i] = upper_range

        return data, self.continuous_columns

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            for i in self.continuous_columns:

                train_low_index = data[i][data[i] < self.lower_range[i]].index
                train_high_index = data[i][data[i] > self.upper_range[i]].index

                data[i][train_low_index] = self.lower_range[i]
                data[i][train_high_index] = self.upper_range[i]

        return data
