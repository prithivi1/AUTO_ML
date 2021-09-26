from Models.Regression.Helper_Script import *
from Models.Cross_Validation_Script import *

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm
import catboost as cb


def final_LinearRegressor(df, y, score):
    model = LinearRegression(n_jobs=-1)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_RidgeRegressor(df, y, score):
    model = Ridge()
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_LassoRegressor(df, y, score):
    model = Lasso()
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_RandomForest(df, y, score, n_estimators, max_depth, min_sample_split, max_leaf_nodes, min_samples_leaf):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_sample_split,
                                  max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf, oob_score=True,
                                  warm_start=True)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_GradientBoosting(df, y, score, learing_rate, n_estimators, max_depth, min_sample_split, max_leaf_nodes,
                           min_samples_leaf):
    model = GradientBoostingRegressor(learning_rate=learing_rate, n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_sample_split, max_leaf_nodes=max_leaf_nodes,
                                      min_samples_leaf=min_samples_leaf)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_xgBoost(df, y, score, xgboost_learing_rate, xgboost_n_estimators, xgboost_max_depth, xgboost_min_child_weight,
                  xgboost_gamma, xgboost_subsample, xgboost_colsample_bytree):
    model = XGBRegressor(learning_rate=xgboost_learing_rate, n_estimators=xgboost_n_estimators,
                         max_depth=xgboost_max_depth, min_child_weight=xgboost_min_child_weight, gamma=xgboost_gamma,
                         subsample=xgboost_subsample, colsample_bytree=xgboost_colsample_bytree)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_lightGBM(df, y, score, lgbm_num_leaves, lgbm_feature_fraction, lgbm_bagging_fraction, lgbm_bagging_freq,
                   lgbm_learning_rate):
    X_train, X_test, y_train, y_test = train_test(df, y)

    train_data = lightgbm.Dataset(X_train, label=y_train)
    test_data = lightgbm.Dataset(X_test, label=y_test)

    parameters = {'num_leaves': lgbm_num_leaves,
                  'feature_fraction': lgbm_feature_fraction,
                  'bagging_fraction': lgbm_bagging_fraction,
                  'bagging_freq': lgbm_bagging_freq,
                  'learning_rate': lgbm_learning_rate,
                  }
    model = lightgbm.train(parameters, train_data, valid_sets=test_data, num_boost_round=5000)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    training_error, testing_error = regression_error_function(y_train, y_test, train_predict, test_predict, score)

    return model, training_error, testing_error


def final_catBoostRegressor(df, y, score, catboost_iterations, catboost_learning_rate, catboost_depth,
                            catboost_l2_leaf_reg):
    X_train, X_test, y_train, y_test = train_test(df, y)

    catBoost_model = cb.CatBoostRegressor(iterations=catboost_iterations, learning_rate=catboost_learning_rate,
                                          depth=catboost_depth, l2_leaf_reg=catboost_l2_leaf_reg)

    catBoost_model.fit(X_train, y_train, eval_set=(X_test, y_test))

    train_predict = catBoost_model.predict(X_train)
    test_predict = catBoost_model.predict(X_test)

    training_error, testing_error = regression_error_function(y_train, y_test, train_predict, test_predict, score)

    return catBoost_model, training_error, testing_error
