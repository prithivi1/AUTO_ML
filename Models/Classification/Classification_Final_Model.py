from Models.Classification.Classification_Helper_Script import *

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def final_logistic_regression(df, y, score, logistic_regression_penalty, logistic_regression_solver,
                              logistic_regression_C):
    model = LogisticRegression(penalty=logistic_regression_penalty, solver=logistic_regression_solver,
                               C=logistic_regression_C, n_jobs=-1)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_KNeighborsClassifier(df, y, score, knn_neighbor, knn_weights, knn_leaf_size):
    model = KNeighborsClassifier(n_neighbors=knn_neighbor, weights=knn_weights, leaf_size=knn_leaf_size)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_decisionTreeClassifier(df, y, score, decision_tree_criterion, decision_tree_splitter, decision_tree_max_depth,
                                 decision_tree_min_sample_split, decision_tree_max_leaf_nodes):
    model = DecisionTreeClassifier(criterion=decision_tree_criterion, splitter=decision_tree_splitter,
                                   max_depth=decision_tree_max_depth, min_samples_split=decision_tree_min_sample_split,
                                   max_leaf_nodes=decision_tree_max_leaf_nodes)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_randomForestClassifier(df, y, score, random_forest_criterion, random_forest_n_estimators,
                                 random_forest_max_depth, random_forest_min_sample_split, random_forest_max_leaf_nodes, random_forest_min_samples_leaf):
    model = RandomForestClassifier(criterion=random_forest_criterion, n_estimators=random_forest_n_estimators,
                                   max_depth=random_forest_max_depth,min_samples_split=random_forest_min_sample_split, min_samples_leaf=random_forest_min_samples_leaf,
                                   max_leaf_nodes=random_forest_max_leaf_nodes)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_adaBoost(df, y, score, adaBoost_n_estimators, adaBoost_learning_rate, adaBoost_algorithm):
    model = AdaBoostClassifier(n_estimators=adaBoost_n_estimators, learning_rate=adaBoost_learning_rate,
                               algorithm=adaBoost_algorithm)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_gradientBoostClassifier(df, y, score, gbBoost_learing_rate, gbBoost_n_estimators, gbBoost_max_depth,
                                  gbBoost_min_sample_split, gbBoost_max_leaf_nodes, gbBoost_min_samples_leaf):
    model = GradientBoostingClassifier(learning_rate=gbBoost_learing_rate, n_estimators=gbBoost_n_estimators,
                                       max_depth=gbBoost_max_depth, min_samples_split=gbBoost_min_sample_split,
                                       max_leaf_nodes=gbBoost_max_leaf_nodes, min_samples_leaf=gbBoost_min_samples_leaf)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_xgBClassifier(df, y, score, xgboost_learing_rate, xgboost_n_estimators, xgboost_max_depth,
                        xgboost_min_child_weight, xgboost_gamma, xgboost_subsample, xgboost_colsample_bytree):
    model = XGBClassifier(learning_rate=xgboost_learing_rate, n_estimators=xgboost_n_estimators,
                          max_depth=xgboost_max_depth, min_child_weight=xgboost_min_child_weight, gamma=xgboost_gamma,
                          subsample=xgboost_subsample, colsample_bytree=xgboost_colsample_bytree)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_lgbmClassifier(df, y, score, lgbm_num_leaves, lgbm_n_estimators, lgbm_max_depth, lgbm_min_child_weight,
                         lgbm_learning_rate):
    model = LGBMClassifier(num_leaves=lgbm_num_leaves, n_estimators=lgbm_n_estimators, max_depth=lgbm_max_depth,
                           min_child_weight=lgbm_min_child_weight, learning_rate=lgbm_learning_rate)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error


def final_catBoostClassifier(df, y, score, catboost_iterations, catboost_learning_rate, catboost_depth,
                             catboost_l2_leaf_reg):
    model = CatBoostClassifier(iterations=catboost_iterations, learning_rate=catboost_learning_rate,
                               depth=catboost_depth, l2_leaf_reg=catboost_l2_leaf_reg)
    model, training_error, testing_error = fitted_model(model, df, y, score)
    return model, training_error, testing_error
