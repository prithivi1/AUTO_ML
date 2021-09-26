import pandas as pd
import scipy.stats as stats
import math


class Cramer_V:

    def __init__(self):
        pass

    @staticmethod
    def cramers_vcorrected_stat(confusion_matrix):
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        number_of_rows = confusion_matrix.sum()
        phi2 = chi2 / number_of_rows
        row, key = confusion_matrix.shape
        phi2_corr = max(0, phi2 - ((key - 1) * (row - 1)) / (number_of_rows - 1))
        r_corr = row - ((row - 1) ** 2) / (number_of_rows - 1)
        k_corr = key - ((key - 1) ** 2) / (number_of_rows - 1)
        return math.sqrt(phi2_corr / min((r_corr - 1), (k_corr - 1)))

    def cramers_V_test(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The input param `{data}` must be of "
                "type pandas.DataFrame"
            )
        else:
            relation = pd.DataFrame(columns=['feature1', 'feature2', 'cramer value'])

            for i in data.columns:
                for j in data.columns:

                    mycrosstab = pd.crosstab(data[i], data[j])
                    mycrosstab = mycrosstab.values

                    cramer_value = Cramer_V.cramers_vcorrected_stat(mycrosstab)

                    if cramer_value < 1:
                        relation.loc[len(relation.index)] = [i, j, cramer_value]

        return relation
