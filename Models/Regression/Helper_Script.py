import lightgbm
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from Models.Cross_Validation_Script import *
from Models.Regression import Base_Model_Script



def train_base_model(X, y, score, n_split):

    base_models = {}

    error_df, final_model = Base_Model_Script.Linear_Regression(X, y, score, n_split)
    base_models[LinearRegression().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = Base_Model_Script.Random_Forest(X, y, score, n_split)
    base_models[RandomForestRegressor().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = Base_Model_Script.Gradient_Boosting(X, y, score, n_split)
    base_models[GradientBoostingRegressor().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = Base_Model_Script.XGBoosting(X, y, score, n_split)
    base_models[XGBRegressor().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = Base_Model_Script.lightBGM(X, y, score, n_split)
    base_models[lightgbm.LGBMModel().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = Base_Model_Script.catBoost(X, y, score, n_split)
    base_models[CatBoostRegressor().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    return base_models


def fitted_model(model, df, y, score):
    model.fit(df, y)
    return model


def manual_cross_validate(model, X, y, score, n_split):
    error_df = pd.DataFrame(columns=['Trail', 'Training Error', 'Testing Error'])

    k_fold = KFold(n_splits=n_split, shuffle=True, random_state=101)
    i = 1
    for train_index, test_index in k_fold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)

        training_error, testing_error = regression_error_function(y_train, y_test, train_prediction, test_prediction,
                                                                  score)
        error_df.loc[len(error_df.index)] = ['CV_' + str(i), training_error, testing_error]
        i = i + 1

    return error_df


def regression_error_function(y_train, y_test, train_prediction, test_prediction, function_name):
    if function_name == 'explained_variance':
        training_error = metrics.explained_variance_score(y_train, train_prediction)
        testing_error = metrics.explained_variance_score(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'max_error':
        training_error = metrics.max_error(y_train, train_prediction)
        testing_error = metrics.max_error(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'neg_mean_absolute_error':
        training_error = metrics.mean_absolute_error(y_train, train_prediction)
        testing_error = metrics.mean_absolute_error(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'neg_mean_squared_error':
        training_error = metrics.mean_squared_error(y_train, train_prediction)
        testing_error = metrics.mean_squared_error(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'neg_root_mean_squared_error':
        training_error = metrics.mean_squared_error(y_train, train_prediction, squared=False)
        testing_error = metrics.mean_squared_error(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'neg_mean_squared_log_error':
        training_error = metrics.mean_squared_log_error(y_train, train_prediction)
        testing_error = metrics.mean_squared_log_error(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'neg_median_absolute_error':
        training_error = metrics.median_absolute_error(y_train, train_prediction)
        testing_error = metrics.median_absolute_error(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'r2':
        training_error = metrics.r2_score(y_train, train_prediction)
        testing_error = metrics.r2_score(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'neg_mean_poisson_deviance':
        training_error = metrics.mean_poisson_deviance(y_train, train_prediction)
        testing_error = metrics.mean_poisson_deviance(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'neg_mean_gamma_deviance':
        training_error = metrics.mean_gamma_deviance(y_train, train_prediction)
        testing_error = metrics.mean_gamma_deviance(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'neg_mean_absolute_percentage_error':
        training_error = metrics.mean_absolute_percentage_error(y_train, train_prediction)
        testing_error = metrics.mean_absolute_percentage_error(y_test, test_prediction)
        return training_error, testing_error
