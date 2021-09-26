class Remove_Unique:

    def __init__(self):
        self.col_to_drop = []

    def fit_transform(self, data):

        for i in data.columns:
            if data[i].nunique() >= len(data[i])-100 or data[i].nunique() == 1:
                self.col_to_drop.append(i)
        data.drop(self.col_to_drop, axis=1, inplace=True)
        return data,self.col_to_drop

    def transform(self, data):
        data.drop(self.col_to_drop, axis=1, inplace=True)
        return data
