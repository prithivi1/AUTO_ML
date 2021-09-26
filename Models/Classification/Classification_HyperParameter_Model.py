from Models.HyperParameter_Script import select_hypertuner_type

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def logistic_regression_grid_search(technique, scoring, cv, logistic_regression_penalty_range,
                                    logistic_regression_solver_range, logistic_regression_C_range, df, y):
    param_test1 = {'penalty': logistic_regression_penalty_range,
                   'solver': logistic_regression_solver_range, 'C': logistic_regression_C_range}

    model = LogisticRegression()

    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def knn_grid_search(technique, scoring, cv, knn_neighbor_range, knn_weights_range, knn_leaf_size_range, df, y):
    param_test1 = {'n_neighbors': knn_neighbor_range,
                   'weights': knn_weights_range, 'leaf_size': knn_leaf_size_range}

    model = KNeighborsClassifier()

    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def dt_grid_search(technique, scoring, cv, decision_tree_criterion_range, decision_tree_splitter,
                   decision_tree_max_depth_range, decision_tree_min_sample_split_range,
                   decision_tree_max_leaf_nodes_range, df, y):
    param_test1 = {'criterion': decision_tree_criterion_range,
                   'splitter': decision_tree_splitter, 'max_depth': decision_tree_max_depth_range,
                   'min_samples_split': decision_tree_min_sample_split_range,
                   'max_leaf_nodes': decision_tree_max_leaf_nodes_range}

    model = DecisionTreeClassifier()

    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def random_forest_grid_search(technique, scoring, cv, random_forest_criterion_range, n_estimators_range,
                              max_depth_range, min_sample_split_range, max_leaf_nodes_range, min_samples_leaf_range, df,
                              y):
    param_test1 = {'criterion': random_forest_criterion_range, 'n_estimators': n_estimators_range,
                   'max_depth': max_depth_range, 'min_samples_split': min_sample_split_range,
                   'max_leaf_nodes': max_leaf_nodes_range, 'min_samples_leaf': min_samples_leaf_range}

    model = RandomForestClassifier(oob_score=True, warm_start=True)

    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def ada_grid_search(technique, scoring, cv, adaBoost_n_estimators_range, adaBoost_learning_rate,
                    adaBoost_algorithm_range, df, y):
    param_test1 = {'n_estimators': adaBoost_n_estimators_range,
                   'learning_rate': adaBoost_learning_rate, 'algorithm': adaBoost_algorithm_range}

    model = AdaBoostClassifier()

    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def gradient_boosting_grid_search(technique, scoring, cv, learning_rate_range, n_estimators_range, max_depth_range,
                                  min_sample_split_range, max_leaf_nodes_range, min_samples_leaf_range, df, y):
    param_test1 = {'learning_rate': learning_rate_range, 'n_estimators': n_estimators_range,
                   'max_depth': max_depth_range, 'min_samples_split': min_sample_split_range,
                   'max_leaf_nodes': max_leaf_nodes_range, 'min_samples_leaf': min_samples_leaf_range}

    model = GradientBoostingClassifier()
    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def XGBRegressor_grid_search(technique, scoring, cv, learning_rate_range, n_estimators_range, max_depth_range,
                             min_child_weight, gamma, subsample, colsample_bytree, df, y):
    param_test1 = {'learning_rate': learning_rate_range, 'n_estimators': n_estimators_range,
                   'max_depth': max_depth_range, 'min_child_weight': min_child_weight, 'gamma': gamma,
                   'subsample': subsample, 'colsample_bytree': colsample_bytree}

    model = XGBClassifier()

    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def LGBMRegressor_grid_search(technique, scoring, cv, lgbm_num_leaves_range, lgbm_n_estimators_range,
                              lgbm_max_depth_range, lgbm_min_child_weight_range, lgbm_learning_rate_range, df, y):
    param_test1 = {'num_leaves': lgbm_num_leaves_range, 'n_estimators': lgbm_n_estimators_range,
                   'max_depth': lgbm_max_depth_range, 'min_child_weight': lgbm_min_child_weight_range,
                   'learing_rate': lgbm_learning_rate_range}

    model = LGBMClassifier()
    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_


def CatBoost_grid_search(technique, scoring, cv, iterations, learning_rate, depth, l2_leaf_reg, df, y):
    param_test1 = {'iterations': iterations, 'learning_rate': learning_rate, 'depth': depth, 'l2_leaf_reg': l2_leaf_reg}

    model = CatBoostClassifier()
    best_params_, cv_results_, best_score_ = select_hypertuner_type(df, y, technique, model, param_test1, scoring, cv)

    return best_params_, cv_results_, best_score_
