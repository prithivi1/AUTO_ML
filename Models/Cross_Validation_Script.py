from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def cv_model(model, x, y, score, n_split):
    k_fold = KFold(n_splits=n_split, shuffle=True, random_state=101)
    error = cross_val_score(model, x, y, cv=k_fold, n_jobs=1, scoring=score)
    print(error)
    print('\n')
    result = {'Model': model, 'Error': abs(round(error.mean(), 4))}
    return result


def train_test(X, y):
    return train_test_split(X, y, test_size=0.20)
