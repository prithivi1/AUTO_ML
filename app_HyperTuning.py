import streamlit as st
import pandas as pd
import numpy as np
import time

from Models.Regression import Regression_HyperParameter_Models
from Models.Classification import Classification_HyperParameter_Model


def regression_hypertuning():
    if 'tune_hyperparameter' not in st.session_state:
        st.session_state.tune_hyperparameter = None

    if 'liner_model_df' not in st.session_state:
        st.session_state.liner_model_df = None

    if 'random_forest_hyper_result' not in st.session_state:
        st.session_state.random_forest_hyper_result = None

    if 'random_forest_cv_results' not in st.session_state:
        st.session_state.random_forest_cv_results = None

    if 'random_forest_best_score' not in st.session_state:
        st.session_state.random_forest_best_score = None

    if 'gbBoost_hyper_result' not in st.session_state:
        st.session_state.gbBoost_hyper_result = None

    if 'gbBoost_cv_results' not in st.session_state:
        st.session_state.gbBoost_cv_results = None

    if 'gbBoost_best_score' not in st.session_state:
        st.session_state.gbBoost_cv_results = None

    if 'xgBoost_hyper_result' not in st.session_state:
        st.session_state.xgBoost_hyper_result = None

    if 'xgBoost_cv_results' not in st.session_state:
        st.session_state.xgBoost_cv_results = None

    if 'xgBoost_best_score' not in st.session_state:
        st.session_state.xgBoost_best_score = None

    if 'lightGBM_hyper_result' not in st.session_state:
        st.session_state.lightGBM_hyper_result = None

    if 'lightGBM_cv_results' not in st.session_state:
        st.session_state.lightGBM_cv_results = None

    if 'lightGBM_best_score' not in st.session_state:
        st.session_state.lightGBM_best_score = None

    if 'catBoost_hyper_result' not in st.session_state:
        st.session_state.catBoost_hyper_result = None

    if 'catBoost_cv_results' not in st.session_state:
        st.session_state.catBoost_cv_results = None

    if 'catBoost_best_score' not in st.session_state:
        st.session_state.catBoost_best_score = None

    # SELECT BEST MODEL FOR HYPER PARAMETER TUNING
    models_lst = []
    if st.session_state.base_models is not None:
        for i in st.session_state.base_models:
            models_lst.append(i)

    model_selected = st.sidebar.selectbox('Select Model For Hyperparamter Tuning :', models_lst)

    # LINEAR MODELS
    if model_selected == 'LinearRegression':
        select_best_model_form = st.sidebar.form(key='best_model')
        select_best_model_form.write('TRAIN RIDGE REGRESSOR')
        select_best_model_form.write('TRAIN LASSO REGRESSOR')
        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'LinearRegression':
        st.session_state.liner_model_df = Regression_HyperParameter_Models.linear_regression_hyper_tuning(
            st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
            st.session_state.n_split_choice)

    if st.session_state.liner_model_df is not None:
        for i in st.session_state.liner_model_df:
            st.subheader(i)
            st.write((st.session_state.liner_model_df[i]['Error']).astype(str))
            st.success('TRAINING AVG MEAN : ' + str(
                sum(st.session_state.liner_model_df[i]['Error']['Training Error']) / len(
                    st.session_state.liner_model_df[i]['Error']['Training Error'])))
            st.success('TESTING AVG ERROR : ' + str(
                sum(st.session_state.liner_model_df[i]['Error']['Testing Error']) / len(
                    st.session_state.liner_model_df[i]['Error']['Testing Error'])))

    if model_selected == 'RandomForestRegressor':
        select_best_model_form = st.sidebar.form(key='best_model')

        random_forest_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                               ['exhaustive_gridSearch', 'exhaustive_randomSearch',
                                                                'halvingGridSearch'])
        random_forest_n_estimators_range = select_best_model_form.multiselect('n_estimators',
                                                                              options=[i for i in range(1, 10000)])
        random_forest_max_depth_range = select_best_model_form.multiselect('max_depth',
                                                                           options=[None] + [i for i in range(1, 20)])
        random_forest_min_sample_split_range = select_best_model_form.multiselect('min_sample_split',
                                                                                  [i for i in range(2, 20)])
        random_forest_max_leaf_nodes_range = select_best_model_form.multiselect('max_leaf_nodes',
                                                                                options=[None] + [i for i in
                                                                                                  range(2, 20)])
        random_forest_min_samples_leaf_range = select_best_model_form.multiselect('min_samples_leaf',
                                                                                  options=[i for i in range(1, 20)])

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'RandomForestRegressor':
        random_forest_start_time = time.time()
        st.session_state.random_forest_hyper_result, st.session_state.random_forest_cv_results, st.session_state.random_forest_best_score = Regression_HyperParameter_Models.random_forest_grid_search(
            random_forest_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice,
            random_forest_n_estimators_range
            , random_forest_max_depth_range, random_forest_min_sample_split_range
            , random_forest_max_leaf_nodes_range, random_forest_min_samples_leaf_range,
            st.session_state.final_df, st.session_state.y)
        random_forest_end_time = time.time()
        st.warning('Time : ' + str((random_forest_end_time - random_forest_start_time) / 60) + ' mins')

    if st.session_state.random_forest_hyper_result is not None:
        st.subheader('HYPERPARAMETER FOR RANDOM FOREST')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.random_forest_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.random_forest_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.random_forest_cv_results))

    if model_selected == 'GradientBoostingRegressor':
        select_best_model_form = st.sidebar.form(key='best_model')

        gbBoost_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                         ['exhaustive_gridSearch', 'exhaustive_randomSearch',
                                                          'halvingGridSearch'])
        gbBoost_learing_rate_range = select_best_model_form.multiselect('learning_rate',
                                                                        options=np.arange(0.1, 1, 0.01))
        gbBoost_n_estimators_range = select_best_model_form.multiselect('n_estimators',
                                                                        options=[i for i in range(1, 10000)])
        gbBoost_max_depth_range = select_best_model_form.multiselect('max_depth',
                                                                     options=[None] + [i for i in range(1, 20)])
        gbBoost_min_sample_split_range = select_best_model_form.multiselect('min_sample_split',
                                                                            options=[i for i in range(2, 20)])
        gbBoost_max_leaf_nodes_range = select_best_model_form.multiselect('mexhaustive_gridSearchax_leaf_nodes',
                                                                          options=[None] + [i for i in range(2, 20)])
        gbBoost_min_samples_leaf_range = select_best_model_form.multiselect('min_samples_leaf',
                                                                            options=[i for i in range(1, 20)])

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'GradientBoostingRegressor':
        gbBoost_start_time = time.time()

        st.session_state.gbBoost_hyper_result, st.session_state.gbBoost_cv_results, st.session_state.gbBoost_best_score = Regression_HyperParameter_Models.gradient_boosting_grid_search(
            gbBoost_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice,
            gbBoost_learing_rate_range,
            gbBoost_n_estimators_range,
            gbBoost_max_depth_range,
            gbBoost_min_sample_split_range, gbBoost_max_leaf_nodes_range,
            gbBoost_min_samples_leaf_range, st.session_state.final_df,
            st.session_state.y)
        gbBoost_end_time = time.time()
        st.warning('Time : ' + str((gbBoost_end_time - gbBoost_start_time) / 60) + ' mins')

    if st.session_state.gbBoost_hyper_result is not None:
        st.subheader('GRID SEARCH RESULTS FOR GRADIENTBOOSTING')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.gbBoost_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.gbBoost_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.gbBoost_cv_results))

    if model_selected == 'XGBRegressor':
        select_best_model_form = st.sidebar.form(key='best_model')

        xgboost_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                         ['exhaustive_gridSearch', 'exhaustive_randomSearch',
                                                          'halvingGridSearch'])
        xgboost_learing_rate_range = select_best_model_form.multiselect('learning_rate',
                                                                        options=np.arange(0.1, 1, 0.01))
        xgboost_n_estimators_range = select_best_model_form.multiselect('n_estimators',
                                                                        options=[i for i in range(1, 10000)])
        xgboost_max_depth_range = select_best_model_form.multiselect('max_depth',
                                                                     options=[None] + [i for i in range(1, 20)])
        xgboost_min_child_weight_range = select_best_model_form.multiselect('min_child_weight',
                                                                            options=[i for i in range(1, 20)])
        xgboost_gamma_range = select_best_model_form.multiselect('gamma', options=np.arange(0.1, 1, 0.01))
        xgboost_subsample_range = select_best_model_form.multiselect('subsample', options=np.arange(0.1, 1, 0.01))
        xgboost_colsample_bytree_range = select_best_model_form.multiselect('colsample_bytree',
                                                                            options=np.arange(0.1, 1, 0.01))

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'XGBRegressor':
        xgBoost_start_time = time.time()

        st.session_state.xgBoost_hyper_result, st.session_state.xgBoost_cv_results, st.session_state.xgBoost_best_score = Regression_HyperParameter_Models.XGBRegressor_grid_search(
            xgboost_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice,
            xgboost_learing_rate_range,
            xgboost_n_estimators_range,
            xgboost_max_depth_range,
            xgboost_min_child_weight_range, xgboost_gamma_range, xgboost_subsample_range
            , xgboost_colsample_bytree_range, st.session_state.final_df,
            st.session_state.y)
        xgBoost_end_time = time.time()
        st.warning('Time : ' + str((xgBoost_end_time - xgBoost_start_time) / 60) + ' mins')

    if st.session_state.xgBoost_hyper_result is not None:
        st.subheader('GRID SEARCH RESULTS FOR XGBRegressor')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.xgBoost_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.xgBoost_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.xgBoost_cv_results))

    if model_selected == 'LGBMModel':
        select_best_model_form = st.sidebar.form(key='best_model')

        lgbm_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                      ['exhaustive_gridSearch', 'exhaustive_randomSearch',
                                                       'halvingGridSearch'])

        lgbm_num_leaves_range = select_best_model_form.multiselect('num_leaves', options=[i for i in range(2, 100)])
        lgbm_feature_fraction_range = select_best_model_form.multiselect('feature_fraction',
                                                                         options=np.arange(0.1, 1, 0.01))
        lgbm_bagging_fraction_range = select_best_model_form.multiselect('bagging_fraction',
                                                                         options=np.arange(0.1, 1, 0.01))
        lgbm_bagging_freq_range = select_best_model_form.multiselect('bagging_freq', options=[i for i in range(1, 100)])
        lgbm_learning_rate_range = select_best_model_form.multiselect('learning_rate', options=np.arange(0.1, 1, 0.01))

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'LGBMModel':
        st.session_state.lightGBM_hyper_result, st.session_state.lightGBM_cv_results, st.session_state.lightGBM_best_score = Regression_HyperParameter_Models.LGBMRegressor_grid_search(
            lgbm_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice, lgbm_num_leaves_range,
            lgbm_feature_fraction_range,
            lgbm_bagging_fraction_range, lgbm_bagging_freq_range, lgbm_learning_rate_range, st.session_state.final_df,
            st.session_state.y)

    if st.session_state.lightGBM_hyper_result is not None:
        st.subheader('GRID SEARCH RESULTS FOR LGBMModel')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.lightGBM_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.lightGBM_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.lightGBM_cv_results))

    if model_selected == 'CatBoostRegressor':
        select_best_model_form = st.sidebar.form(key='best_model')

        catboost_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                          ['exhaustive_gridSearch', 'exhaustive_randomSearch',
                                                           'halvingGridSearch'])

        catboost_iterations_range = select_best_model_form.multiselect('iterations',
                                                                       options=[i for i in range(1000, 10000)])
        catboost_learning_rate_range = select_best_model_form.multiselect('learning_rate',
                                                                          options=np.arange(0.1, 1, 0.001))
        catboost_depth_range = select_best_model_form.multiselect('depth', options=[i for i in range(1, 50)])
        catboost_l2_leaf_reg_range = select_best_model_form.multiselect('l2_leaf_reg',
                                                                        options=[i for i in range(1, 100)])

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'CatBoostRegressor':
        st.session_state.catBoost_hyper_result, st.session_state.catBoost_cv_results, st.session_state.catBoost_best_score = Regression_HyperParameter_Models.CatBoost_grid_search(
            catboost_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice,
            catboost_iterations_range,
            catboost_learning_rate_range, catboost_depth_range,
            catboost_l2_leaf_reg_range, st.session_state.final_df, st.session_state.y)

    if st.session_state.catBoost_hyper_result is not None:
        st.subheader('GRID SEARCH RESULTS FOR CATBOOSTRegressor')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.catBoost_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.catBoost_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.catBoost_cv_results))


def classification_hypertuning():
    if 'tune_hyperparameter' not in st.session_state:
        st.session_state.tune_hyperparameter = None

    if 'logistic_regression_hyper_result' not in st.session_state:
        st.session_state.logistic_regression_hyper_result = None

    if 'logistic_regression_cv_results' not in st.session_state:
        st.session_state.logistic_regression_cv_results = None

    if 'logistic_regression_best_score' not in st.session_state:
        st.session_state.logistic_regression_best_score = None

    if 'knn_hyper_result' not in st.session_state:
        st.session_state.knn_hyper_result = None

    if 'knn_cv_results' not in st.session_state:
        st.session_state.knn_cv_results = None

    if 'knn_best_score' not in st.session_state:
        st.session_state.knn_best_score = None

    if 'decision_tree_hyper_result' not in st.session_state:
        st.session_state.decision_tree_hyper_result = None

    if 'decision_tree_cv_results' not in st.session_state:
        st.session_state.decision_tree_cv_results = None

    if 'decision_tree_best_score' not in st.session_state:
        st.session_state.decision_tree_best_score = None

    if 'random_forest_hyper_result' not in st.session_state:
        st.session_state.random_forest_hyper_result = None

    if 'random_forest_cv_results' not in st.session_state:
        st.session_state.random_forest_cv_results = None

    if 'random_forest_best_score' not in st.session_state:
        st.session_state.random_forest_best_score = None

    if 'adaBoost_hyper_result' not in st.session_state:
        st.session_state.adaBoost_hyper_result = None

    if 'adaBoost_cv_results' not in st.session_state:
        st.session_state.adaBoost_cv_results = None

    if 'adaBoost_best_score' not in st.session_state:
        st.session_state.adaBoost_cv_results = None

    if 'gbBoost_hyper_result' not in st.session_state:
        st.session_state.gbBoost_hyper_result = None

    if 'gbBoost_cv_results' not in st.session_state:
        st.session_state.gbBoost_cv_results = None

    if 'gbBoost_best_score' not in st.session_state:
        st.session_state.gbBoost_cv_results = None

    if 'xgBoost_hyper_result' not in st.session_state:
        st.session_state.xgBoost_hyper_result = None

    if 'xgBoost_cv_results' not in st.session_state:
        st.session_state.xgBoost_cv_results = None

    if 'xgBoost_best_score' not in st.session_state:
        st.session_state.xgBoost_best_score = None

    if 'lightGBM_hyper_result' not in st.session_state:
        st.session_state.lightGBM_hyper_result = None

    if 'lightGBM_cv_results' not in st.session_state:
        st.session_state.lightGBM_cv_results = None

    if 'lightGBM_best_score' not in st.session_state:
        st.session_state.lightGBM_best_score = None

    if 'catBoost_hyper_result' not in st.session_state:
        st.session_state.catBoost_hyper_result = None

    if 'catBoost_cv_results' not in st.session_state:
        st.session_state.catBoost_cv_results = None

    if 'catBoost_best_score' not in st.session_state:
        st.session_state.catBoost_best_score = None

    models_lst = []
    if st.session_state.base_models is not None:
        for i in st.session_state.base_models:
            models_lst.append(i)

    model_selected = st.sidebar.selectbox('Select Model For Hyperparamter Tuning :', models_lst)

    if model_selected == 'LogisticRegression':
        select_best_model_form = st.sidebar.form(key='best_model')

        logistic_regression_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                                     ['exhaustive_gridSearch',
                                                                      'exhaustive_randomSearch',
                                                                      'halvingGridSearch'])

        logistic_regression_penalty_range = select_best_model_form.multiselect('penalty',
                                                                               options=['l2', 'l1', 'elasticnet',
                                                                                        'none'], default='l2')

        logistic_regression_solver_range = select_best_model_form.multiselect('solver', options=['lbfgs', 'newton-cg',
                                                                                                 'liblinear', 'sag',
                                                                                                 'saga'],
                                                                              default='lbfgs')

        logistic_regression_C_range = select_best_model_form.multiselect('c_value',
                                                                         options=[i for i in np.arange(1.0, 100, 0.01)],
                                                                         default=1)

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'LogisticRegression':
        logistic_regression_start_time = time.time()
        st.session_state.logistic_regression_hyper_result, st.session_state.logistic_regression_cv_results, st.session_state.logistic_regression_best_score = Classification_HyperParameter_Model.logistic_regression_grid_search(
            logistic_regression_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice,
            logistic_regression_penalty_range, logistic_regression_solver_range, logistic_regression_C_range,
            st.session_state.final_df, st.session_state.y)
        logistic_regression_stop_time = time.time()
        st.warning('Time : ' + str((logistic_regression_stop_time - logistic_regression_start_time) / 60) + ' mins')

    if st.session_state.logistic_regression_hyper_result is not None:
        st.subheader('HYPERPARAMETER FOR LOGISTIC REGRESSION')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.logistic_regression_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.logistic_regression_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.logistic_regression_cv_results))

    if model_selected == 'KNeighborsClassifier':
        select_best_model_form = st.sidebar.form(key='best_model')

        knn_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                     ['exhaustive_gridSearch',
                                                      'exhaustive_randomSearch',
                                                      'halvingGridSearch'])

        knn_neighbor_range = select_best_model_form.multiselect('n_neighbors', options=[i for i in range(1, 10000, 2)],
                                                                default=5)

        knn_weights_range = select_best_model_form.multiselect('weights', options=['uniform', 'distance'],
                                                               default='uniform')

        knn_leaf_size_range = select_best_model_form.multiselect('leaf_size', options=[i for i in range(1, 1000)],
                                                                 default=30)

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'KNeighborsClassifier':
        knn_start_time = time.time()
        st.session_state.knn_hyper_result, st.session_state.knn_cv_results, st.session_state.knn_best_score = Classification_HyperParameter_Model.knn_grid_search(
            knn_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice, knn_neighbor_range, knn_weights_range,
            knn_leaf_size_range, st.session_state.final_df, st.session_state.y)
        knn_stop_time = time.time()
        st.warning('Time : ' + str((knn_stop_time - knn_start_time) / 60) + ' mins')

    if st.session_state.knn_hyper_result is not None:
        st.subheader('HYPERPARAMETER FOR KNN')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.knn_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.knn_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.knn_cv_results))

    if model_selected == 'DecisionTreeClassifier':
        select_best_model_form = st.sidebar.form(key='best_model')

        decision_tree_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                               ['exhaustive_gridSearch',
                                                                'exhaustive_randomSearch',
                                                                'halvingGridSearch'])
        decision_tree_criterion_range = select_best_model_form.multiselect('criterion', options=['gini', 'entropy'],
                                                                           default='gini')

        decision_tree_splitter = select_best_model_form.multiselect('splitter', options=['best', 'random'],
                                                                    default='best')

        decision_tree_max_depth_range = select_best_model_form.multiselect('max_depth',
                                                                           options=[None] + [i for i in range(1, 20)],
                                                                           default=None)

        decision_tree_min_sample_split_range = select_best_model_form.multiselect('min_sample_split',
                                                                                  options=[i for i in range(2, 20)], default=2)

        decision_tree_max_leaf_nodes_range = select_best_model_form.multiselect('max_leaf_nodes',
                                                                                options=[None] + [i for i in
                                                                                                  range(2, 20)], default=None)

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'DecisionTreeClassifier':
        dt_start_time = time.time()
        st.session_state.decision_tree_hyper_result, st.session_state.decision_tree_cv_results, st.session_state.decision_tree_best_score = Classification_HyperParameter_Model.dt_grid_search(
            decision_tree_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice,
            decision_tree_criterion_range, decision_tree_splitter, decision_tree_max_depth_range,
            decision_tree_min_sample_split_range, decision_tree_max_leaf_nodes_range, st.session_state.final_df,
            st.session_state.y)
        dt_stop_time = time.time()
        st.warning('Time : ' + str((dt_stop_time - dt_start_time) / 60) + ' mins')

    if st.session_state.decision_tree_hyper_result is not None:
        st.subheader('HYPERPARAMETER FOR DECISION TREE')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.decision_tree_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.decision_tree_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.decision_tree_cv_results))

    if model_selected == 'RandomForestClassifier':
        select_best_model_form = st.sidebar.form(key='best_model')

        random_forest_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                               ['exhaustive_gridSearch', 'exhaustive_randomSearch',
                                                                'halvingGridSearch'])

        random_forest_criterion_range = select_best_model_form.multiselect('criterion', options=['gini', 'entropy'],
                                                                           default='gini')

        random_forest_n_estimators_range = select_best_model_form.multiselect('n_estimators',
                                                                              options=[i for i in range(1, 10000)],
                                                                              default=100)
        random_forest_max_depth_range = select_best_model_form.multiselect('max_depth',
                                                                           options=[None] + [i for i in range(1, 20)],
                                                                           default=None)
        random_forest_min_sample_split_range = select_best_model_form.multiselect('min_sample_split',
                                                                                  options=[i for i in range(2, 20)], default=2)
        random_forest_max_leaf_nodes_range = select_best_model_form.multiselect('max_leaf_nodes',
                                                                                options=[None] + [i for i in
                                                                                                  range(2, 20)], default=None)
        random_forest_min_samples_leaf_range = select_best_model_form.multiselect('min_samples_leaf',
                                                                                  options=[i for i in range(1, 20)], default=1)

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'RandomForestClassifier':
        random_forest_start_time = time.time()
        st.session_state.random_forest_hyper_result, st.session_state.random_forest_cv_results, st.session_state.random_forest_best_score = Classification_HyperParameter_Model.random_forest_grid_search(
            random_forest_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice,
            random_forest_criterion_range, random_forest_n_estimators_range, random_forest_max_depth_range,
            random_forest_min_sample_split_range
            , random_forest_max_leaf_nodes_range, random_forest_min_samples_leaf_range, st.session_state.final_df,
            st.session_state.y)
        random_forest_end_time = time.time()
        st.warning('Time : ' + str((random_forest_end_time - random_forest_start_time) / 60) + ' mins')

    if st.session_state.random_forest_hyper_result is not None:
        st.subheader('HYPERPARAMETER FOR RANDOM FOREST')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.random_forest_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.random_forest_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.random_forest_cv_results))

    if model_selected == 'AdaBoostClassifier':
        select_best_model_form = st.sidebar.form(key='best_model')

        adaBoost_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                          ['exhaustive_gridSearch', 'exhaustive_randomSearch',
                                                           'halvingGridSearch'])

        adaBoost_n_estimators_range = select_best_model_form.multiselect('n_estimators',
                                                                         options=[i for i in range(1, 10000)],
                                                                         default=100)
        adaBoost_learning_rate = select_best_model_form.multiselect('learning_rate', options=np.arange(0, 10, 0.01),
                                                                    default=1)
        adaBoost_algorithm_range = select_best_model_form.multiselect('algorithm', options=['SAMME', 'SAMME.R'],
                                                                      default='SAMME.R')

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'AdaBoostClassifier':
        adaBoost_start_time = time.time()
        st.session_state.adaBoost_hyper_result, st.session_state.adaBoost_cv_results, st.session_state.adaBoost_best_score = Classification_HyperParameter_Model.ada_grid_search(
            adaBoost_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice, adaBoost_n_estimators_range,
            adaBoost_learning_rate, adaBoost_algorithm_range, st.session_state.final_df, st.session_state.y)
        adaBoost_end_time = time.time()
        st.warning('Time : ' + str((adaBoost_end_time - adaBoost_start_time) / 60) + ' mins')

    if st.session_state.adaBoost_hyper_result is not None:
        st.subheader('HYPERPARAMETER FOR ADABOOST')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.adaBoost_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.adaBoost_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.adaBoost_cv_results))

    if model_selected == 'GradientBoostingClassifier':
        select_best_model_form = st.sidebar.form(key='best_model')

        gbBoost_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                         ['exhaustive_gridSearch', 'exhaustive_randomSearch',
                                                          'halvingGridSearch'])

        gbBoost_learing_rate_range = select_best_model_form.multiselect('learning_rate', options=np.arange(0, 1, 0.01), default=0.1)
        gbBoost_n_estimators_range = select_best_model_form.multiselect('n_estimators',
                                                                        options=[i for i in range(1, 10000)], default=100)
        gbBoost_max_depth_range = select_best_model_form.multiselect('max_depth',
                                                                     options=[None] + [i for i in range(1, 20)], default=3)
        gbBoost_min_sample_split_range = select_best_model_form.multiselect('min_sample_split',
                                                                            options=[i for i in range(2, 20)], default=2)
        gbBoost_max_leaf_nodes_range = select_best_model_form.multiselect('max_leaf_nodes',
                                                                          options=[None] + [i for i in range(2, 20)], default=None)
        gbBoost_min_samples_leaf_range = select_best_model_form.multiselect('min_samples_leaf',
                                                                            options=[i for i in range(1, 20)], default=1)

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'GradientBoostingClassifier':
        gbBoost_start_time = time.time()

        st.session_state.gbBoost_hyper_result, st.session_state.gbBoost_cv_results, st.session_state.gbBoost_best_score = Classification_HyperParameter_Model.gradient_boosting_grid_search(
            gbBoost_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice, gbBoost_learing_rate_range,
            gbBoost_n_estimators_range,
            gbBoost_max_depth_range, gbBoost_min_sample_split_range, gbBoost_max_leaf_nodes_range,
            gbBoost_min_samples_leaf_range, st.session_state.final_df,
            st.session_state.y)
        gbBoost_end_time = time.time()
        st.warning('Time : ' + str((gbBoost_end_time - gbBoost_start_time) / 60) + ' mins')

    if st.session_state.gbBoost_hyper_result is not None:
        st.subheader('GRID SEARCH RESULTS FOR GRADIENTBOOSTING')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.gbBoost_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.gbBoost_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.gbBoost_cv_results))

    if model_selected == 'XGBClassifier':
        select_best_model_form = st.sidebar.form(key='best_model')

        xgboost_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                         ['exhaustive_gridSearch', 'exhaustive_randomSearch',
                                                          'halvingGridSearch'])
        xgboost_learing_rate_range = select_best_model_form.multiselect('learning_rate', options=np.arange(0.1, 1, 0.01), default=0.1)
        xgboost_n_estimators_range = select_best_model_form.multiselect('n_estimators',
                                                                        options=[i for i in range(1, 10000)], default=100)
        xgboost_max_depth_range = select_best_model_form.multiselect('max_depth',
                                                                     options=[None] + [i for i in range(1, 20)], default=None)
        xgboost_min_child_weight_range = select_best_model_form.multiselect('min_child_weight',
                                                                            options=[i for i in range(1, 20)], default=1)
        xgboost_gamma_range = select_best_model_form.multiselect('gamma', options=np.arange(0.1, 1, 0.01), default=0.1)
        xgboost_subsample_range = select_best_model_form.multiselect('subsample', options=np.arange(0.1, 1, 0.01),default=0.1)
        xgboost_colsample_bytree_range = select_best_model_form.multiselect('colsample_bytree',
                                                                            options=np.arange(0.1, 1, 0.01), default=0.1)

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'XGBClassifier':
        xgBoost_start_time = time.time()

        st.session_state.xgBoost_hyper_result, st.session_state.xgBoost_cv_results, st.session_state.xgBoost_best_score = Classification_HyperParameter_Model.XGBRegressor_grid_search(
            xgboost_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice, xgboost_learing_rate_range,
            xgboost_n_estimators_range,
            xgboost_max_depth_range,
            xgboost_min_child_weight_range, xgboost_gamma_range, xgboost_subsample_range
            , xgboost_colsample_bytree_range, st.session_state.final_df,
            st.session_state.y)
        xgBoost_end_time = time.time()
        st.warning('Time : ' + str((xgBoost_end_time - xgBoost_start_time) / 60) + ' mins')

    if st.session_state.xgBoost_hyper_result is not None:
        st.subheader('GRID SEARCH RESULTS FOR XGBClassifier')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.xgBoost_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.xgBoost_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.xgBoost_cv_results))

    if model_selected == 'LGBMModel':
        select_best_model_form = st.sidebar.form(key='best_model')

        lgbm_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                      ['exhaustive_gridSearch', 'exhaustive_randomSearch',
                                                       'halvingGridSearch'])

        lgbm_num_leaves_range = select_best_model_form.multiselect('num_leaves', options=[i for i in range(2, 100)], default=2)
        lgbm_n_estimators_range = select_best_model_form.multiselect('n_estimators',
                                                                     options=[i for i in range(1, 10000)], default=100)
        lgbm_max_depth_range = select_best_model_form.multiselect('max_depth',
                                                                  options=[None] + [i for i in range(1, 20)], default=None)
        lgbm_min_child_weight_range = select_best_model_form.multiselect('min_child_weight',
                                                                         options=[i for i in range(1, 20)], default=1)
        lgbm_learning_rate_range = select_best_model_form.multiselect('learning_rate', options=np.arange(0.1, 1, 0.01), default=0.1)

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'LGBMModel':
        st.session_state.lightGBM_hyper_result, st.session_state.lightGBM_cv_results, st.session_state.lightGBM_best_score = Classification_HyperParameter_Model.LGBMRegressor_grid_search(
            lgbm_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice, lgbm_num_leaves_range,
            lgbm_n_estimators_range,
            lgbm_max_depth_range, lgbm_min_child_weight_range, lgbm_learning_rate_range, st.session_state.final_df,
            st.session_state.y)

    if st.session_state.lightGBM_hyper_result is not None:
        st.subheader('GRID SEARCH RESULTS FOR LGBMModel')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.lightGBM_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.lightGBM_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.lightGBM_cv_results))

    if model_selected == 'CatBoostClassifier':
        select_best_model_form = st.sidebar.form(key='best_model')

        catboost_tuner = select_best_model_form.selectbox('Select tuning Technique',
                                                          ['exhaustive_gridSearch', 'exhaustive_randomSearch',
                                                           'halvingGridSearch'])

        catboost_iterations_range = select_best_model_form.multiselect('iterations',
                                                                       options=[i for i in range(1000, 10000)], default=1000)
        catboost_learning_rate_range = select_best_model_form.multiselect('learning_rate',
                                                                          options=np.arange(0.1, 1, 0.001), default=0.1)
        catboost_depth_range = select_best_model_form.multiselect('depth', options=[i for i in range(1, 50)], default=1)
        catboost_l2_leaf_reg_range = select_best_model_form.multiselect('l2_leaf_reg',
                                                                        options=[i for i in range(1, 100)],default=1)

        st.session_state.tune_hyperparameter = select_best_model_form.form_submit_button('HYPERPARAMETER TUNING')

    if st.session_state.tune_hyperparameter and model_selected == 'CatBoostClassifier':
        st.session_state.catBoost_hyper_result, st.session_state.catBoost_cv_results, st.session_state.catBoost_best_score = Classification_HyperParameter_Model.CatBoost_grid_search(
            catboost_tuner, st.session_state.error_function_choice, st.session_state.n_split_choice, catboost_iterations_range,
            catboost_learning_rate_range, catboost_depth_range,
            catboost_l2_leaf_reg_range, st.session_state.final_df, st.session_state.y)

    if st.session_state.catBoost_hyper_result is not None:
        st.subheader('GRID SEARCH RESULTS FOR CATBOOST')
        st.subheader('BEST PARAMETERS')
        st.write(st.session_state.catBoost_hyper_result)
        st.subheader('BEST SCORE')
        st.write(st.session_state.catBoost_best_score)
        st.subheader('CV RESULT')
        st.write(pd.DataFrame.from_dict(st.session_state.catBoost_cv_results))
