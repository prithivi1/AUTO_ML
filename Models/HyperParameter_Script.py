from sklearn.experimental import enable_halving_search_cv

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import HalvingGridSearchCV


def select_hypertuner_type(df, y, technique, model, param, score, n_split):
    if technique == 'exhaustive_gridSearch':
        return exhaustive_gridSearch(model, param, score, n_split, df, y)

    elif technique == 'exhaustive_randomSearch':
        return exhaustive_randomSearch(model, param, score, n_split, df, y)

    elif technique == 'halvingGridSearch':
        return halvingGridSearch(model, param, score, n_split, df, y)


def exhaustive_gridSearch(estimator, param_grid, scoring, cv, df, y):
    gsearch1 = GridSearchCV(estimator=estimator, param_grid=param_grid,
                            scoring=scoring, n_jobs=-1, cv=cv, return_train_score=True,
                            error_score='raise')
    gsearch1.fit(df, y)
    return gsearch1.best_params_, gsearch1.cv_results_, gsearch1.best_score_


def exhaustive_randomSearch(estimator, param_grid, scoring, cv, df, y, n_iteration=100):
    gsearch1 = RandomizedSearchCV(n_iter=n_iteration, estimator=estimator, param_grid=param_grid,
                                  scoring=scoring, n_jobs=-1, cv=cv, return_train_score=True,
                                  error_score='raise')
    gsearch1.fit(df, y)
    return gsearch1.best_params_, gsearch1.cv_results_, gsearch1.best_score_


def halvingGridSearch(estimator, param_grid, scoring, cv, df, y):
    gsearch1 = HalvingGridSearchCV(estimator=estimator, param_grid=param_grid,
                                   scoring=scoring, n_jobs=-1, cv=cv, return_train_score=True,
                                   error_score='raise')
    gsearch1.fit(df, y)
    return gsearch1.best_params_, gsearch1.cv_results_, gsearch1.best_score_
