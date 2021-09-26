import pandas as pd
import scipy.stats as stats


class Annova:

    def __init__(self):
        self.annova_categorical_features_to_drop = []

    def fit_transform(self, data, target_column, threshold=0.05):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            annova_data = pd.DataFrame(columns=['feature1', 'p-value'])

            for i in data.columns:
                lst = list(data[i].value_counts().index)
                dataframe = pd.concat([data, target_column], axis=1)
                a = dataframe.groupby(by=i)

                var1 = [list(dataframe.iloc[a.indices[val]][target_column.name]) for val in lst]

                f_score, p_value = stats.f_oneway(*var1)

                if p_value < threshold:
                    annova_data.loc[len(annova_data.index)] = [i, p_value]
                else:
                    self.annova_categorical_features_to_drop.append(i)
                    data.drop(i, axis=1, inplace=True)

        return data, annova_data, self.annova_categorical_features_to_drop

    def tranform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            if len(self.annova_categorical_features_to_drop) == 0:
                return data
            else:
                data.drop(self.annova_categorical_features_to_drop, axis=1, inplace=True)
            return data
