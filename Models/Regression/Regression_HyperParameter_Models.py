from Models.HyperParameter_Script import select_hypertuner_type

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from Models.Regression.Base_Model_Script import ridgeRegressor, lassoRegressor


def linear_regression_hyper_tuning(X, y, score, n_split):
    ml_models = {}

    error_df, final_model = ridgeRegressor(X, y, score, n_split)
    ml_models[Ridge().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = lassoRegressor(X, y, score, n_split)
    ml_models[Lasso().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    return ml_models


def random_forest_grid_search(technique, scoring, cv, n_estimators_range, max_depth_range, min_sample_split_range,
                              max_leaf_nodes_range, min_samples_leaf_range, df, y):
    param_test1 = {'n_estimators': n_estimators_range,
                   'max_depth': max_depth_range, 'min_samples_split': min_sample_split_range,
                   'max_leaf_nodes': max_leaf_nodes_range, 'min_samples_leaf': min_samples_leaf_range}

    model = RandomForestRegressor(oob_score=True, warm_start=True)

    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def gradient_boosting_grid_search(technique, scoring, cv, learning_rate_range, n_estimators_range, max_depth_range,
                                  min_sample_split_range, max_leaf_nodes_range, min_samples_leaf_range, df, y):
    param_test1 = {'learning_rate': learning_rate_range, 'n_estimators': n_estimators_range,
                   'max_depth': max_depth_range, 'min_samples_split': min_sample_split_range,
                   'max_leaf_nodes': max_leaf_nodes_range, 'min_samples_leaf': min_samples_leaf_range}

    model = GradientBoostingRegressor(warm_start=True)

    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def XGBRegressor_grid_search(technique, scoring, cv, learning_rate_range, n_estimators_range, max_depth_range,
                             min_child_weight, gamma, subsample, colsample_bytree, df, y):
    param_test1 = {'learning_rate': learning_rate_range, 'n_estimators': n_estimators_range,
                   'max_depth': max_depth_range, 'min_child_weight': min_child_weight, 'gamma': gamma,
                   'subsample': subsample, 'colsample_bytree': colsample_bytree}

    model = XGBRegressor()
    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def LGBMRegressor_grid_search(technique, scoring, cv, num_leaves, feature_fraction, bagging_fraction, bagging_freq,
                              learing_rate, df, y):
    param_test1 = {'num_leaves': num_leaves, 'feature_fraction': feature_fraction, 'bagging_fraction': bagging_fraction,
                   'bagging_freq': bagging_freq, 'learing_rate': learing_rate}

    model = LGBMRegressor()

    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def CatBoost_grid_search(technique, scoring, cv, iterations, learning_rate, depth, l2_leaf_reg, df, y):
    param_test1 = {'iterations': iterations, 'learning_rate': learning_rate, 'depth': depth, 'l2_leaf_reg': l2_leaf_reg}
    model = CatBoostRegressor()

    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_
