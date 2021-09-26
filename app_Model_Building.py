import streamlit as st
import numpy as np

from Models.Classification import Classification_Final_Model
from Models.Regression import Final_Model_Script


def regression_model_building():
    models_lst = []
    if st.session_state.base_models is not None:
        for i in st.session_state.base_models:
            models_lst.append(i)

    select_train_model = st.sidebar.selectbox('Select Model For Training :', models_lst)

    if select_train_model == 'LinearRegression':
        select_training_model = st.sidebar.form(key='train_model')
        linear_model_type = select_training_model.selectbox('Select Linear Model',
                                                            ['LinearRegressor', 'RidgeRegressor', 'LassoRegressor'])
        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'LinearRegression' and st.session_state.train_model_button:

            if linear_model_type == 'LinearRegressor':
                st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Final_Model_Script.final_LinearRegressor(
                    st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice)
            elif linear_model_type == 'RidgeRegressor':
                st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Final_Model_Script.final_RidgeRegressor(
                    st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice)
            elif linear_model_type == 'LassoRegressor':
                st.session_state.final_trained_model, st.sesstsion_state.final_training_error, st.session_state.final_testing_error = Final_Model_Script.final_LassoRegressor(
                    st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice)

    if select_train_model == 'RandomForestRegressor':
        select_training_model = st.sidebar.form(key='train_model')

        random_forest_n_estimators = select_training_model.selectbox('n_estimators',
                                                                     options=[i for i in range(1, 10000)])
        random_forest_max_depth = select_training_model.selectbox('max_depth',
                                                                  options=[None] + [i for i in range(1, 20)])
        random_forest_min_sample_split = select_training_model.selectbox('min_sample_split', [i for i in range(2, 20)])
        random_forest_max_leaf_nodes = select_training_model.selectbox('max_leaf_nodes',
                                                                       options=[None] + [i for i in range(2, 20)])
        random_forest_min_samples_leaf = select_training_model.selectbox('min_samples_leaf',
                                                                         options=[i for i in range(1, 20)])

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'RandomForestRegressor' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Final_Model_Script.final_RandomForest(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
                random_forest_n_estimators,
                random_forest_max_depth, random_forest_min_sample_split, random_forest_max_leaf_nodes,
                random_forest_min_samples_leaf)

    if select_train_model == 'GradientBoostingRegressor':
        select_training_model = st.sidebar.form(key='train_model')

        gbBoost_learing_rate = select_training_model.selectbox('learning_rate', options=np.arange(0.1, 1, 0.01))
        gbBoost_n_estimators = select_training_model.selectbox('n_estimators', options=[i for i in range(1, 10000)])
        gbBoost_max_depth = select_training_model.selectbox('max_depth', options=[None] + [i for i in range(1, 20)])
        gbBoost_min_sample_split = select_training_model.selectbox('min_sample_split',
                                                                   options=[i for i in range(2, 20)])
        gbBoost_max_leaf_nodes = select_training_model.selectbox('max_leaf_nodes',
                                                                 options=[None] + [i for i in range(2, 20)])
        gbBoost_min_samples_leaf = select_training_model.selectbox('min_samples_leaf',
                                                                   options=[i for i in range(1, 20)])

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'GradientBoostingRegressor' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Final_Model_Script.final_GradientBoosting(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
                gbBoost_learing_rate,
                gbBoost_n_estimators, gbBoost_max_depth, gbBoost_min_sample_split, gbBoost_max_leaf_nodes,
                gbBoost_min_samples_leaf)

    if select_train_model == 'XGBRegressor':
        select_training_model = st.sidebar.form(key='train_model')

        xgboost_learing_rate = select_training_model.selectbox('learning_rate', options=np.arange(0.1, 1, 0.01))
        xgboost_n_estimators = select_training_model.selectbox('n_estimators', options=[i for i in range(1, 10000)])
        xgboost_max_depth = select_training_model.selectbox('max_depth', options=[None] + [i for i in range(1, 20)])
        xgboost_min_child_weight = select_training_model.selectbox('min_child_weight',
                                                                   options=[i for i in range(1, 20)])
        xgboost_gamma = select_training_model.selectbox('gamma', options=np.arange(0.1, 1, 0.01))
        xgboost_subsample = select_training_model.selectbox('subsample', options=np.arange(0.1, 1, 0.01))
        xgboost_colsample_bytree = select_training_model.selectbox('colsample_bytree', options=np.arange(0.1, 1, 0.01))

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'XGBRegressor' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Final_Model_Script.final_xgBoost(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
                xgboost_learing_rate,
                xgboost_n_estimators, xgboost_max_depth, xgboost_min_child_weight, xgboost_gamma, xgboost_subsample,
                xgboost_colsample_bytree)

    if select_train_model == 'LGBMModel':
        select_training_model = st.sidebar.form(key='train_model')

        lgbm_num_leaves = select_training_model.selectbox('num_leaves', options=[i for i in range(1, 100)])
        lgbm_feature_fraction = select_training_model.selectbox('feature_fraction', options=np.arange(0.1, 1, 0.01))
        lgbm_bagging_fraction = select_training_model.selectbox('bagging_fraction', options=np.arange(0.1, 1, 0.01))
        lgbm_bagging_freq = select_training_model.selectbox('bagging_freq', options=[i for i in range(1, 100)])
        lgbm_learning_rate_range = select_training_model.selectbox('learning_rate', options=np.arange(0.1, 1, 0.01))

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'LGBMModel' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Final_Model_Script.final_lightGBM(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice, lgbm_num_leaves,
                lgbm_feature_fraction, lgbm_bagging_fraction, lgbm_bagging_freq, lgbm_learning_rate_range)

    if select_train_model == 'CatBoostRegressor':
        select_training_model = st.sidebar.form(key='train_model')

        catboost_iterations = select_training_model.selectbox('iterations', options=[i for i in range(1000, 10000)])
        catboost_learning_rate = select_training_model.selectbox('learning_rate', options=np.arange(0.1, 1, 0.001))
        catboost_depth = select_training_model.selectbox('depth', options=[i for i in range(1, 50)])
        catboost_l2_leaf_reg = select_training_model.selectbox('l2_leaf_reg', options=[i for i in range(1, 100)])

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'CatBoostRegressor' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Final_Model_Script.final_catBoostRegressor(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
                catboost_iterations,
                catboost_learning_rate, catboost_depth, catboost_l2_leaf_reg)

    if st.session_state.final_trained_model is not None:
        st.subheader('MODEL TRAINED : ')
        st.success(st.session_state.final_trained_model)

    return st.session_state.final_trained_model


def classification_model_building():
    models_lst = []
    if st.session_state.base_models is not None:
        for i in st.session_state.base_models:
            models_lst.append(i)

    select_train_model = st.sidebar.selectbox('Select Model For Training :', models_lst)

    if select_train_model == 'LogisticRegression':
        select_training_model = st.sidebar.form(key='train_model')

        logistic_regression_penalty = select_training_model.selectbox('penalty', ['l2', 'l1', 'elasticnet', 'none'])

        logistic_regression_solver = select_training_model.selectbox('solver',
                                                                     ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'])

        logistic_regression_C = select_training_model.selectbox('c_value', [i for i in np.arange(1.0, 100, 0.01)])

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'LogisticRegression' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Classification_Final_Model.final_logistic_regression(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
                logistic_regression_penalty,
                logistic_regression_solver, logistic_regression_C)

    if select_train_model == 'KNeighborsClassifier':
        select_training_model = st.sidebar.form(key='train_model')

        knn_neighbor = select_training_model.selectbox('n_neighbors', [i for i in range(1, 10000, 2)])

        knn_weights = select_training_model.selectbox('weights', ['uniform', 'distance'])

        knn_leaf_size = select_training_model.selectbox('leaf_size', [i for i in range(1, 1000)])

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'KNeighborsClassifier' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Classification_Final_Model.final_KNeighborsClassifier(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice, knn_neighbor,
                knn_weights,
                knn_leaf_size)

    if select_train_model == 'DecisionTreeClassifier':
        select_training_model = st.sidebar.form(key='train_model')

        decision_tree_criterion = select_training_model.selectbox('criterion', ['gini', 'entropy'])

        decision_tree_splitter = select_training_model.selectbox('splitter', ['best', 'random'])

        decision_tree_max_depth = select_training_model.selectbox('max_depth', [None] + [i for i in range(1, 20)])

        decision_tree_min_sample_split = select_training_model.selectbox('min_sample_split', [i for i in range(2, 20)])

        decision_tree_max_leaf_nodes = select_training_model.selectbox('max_leaf_nodes',
                                                                       options=[None] + [i for i in range(2, 20)])

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'DecisionTreeClassifier' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Classification_Final_Model.final_decisionTreeClassifier(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
                decision_tree_criterion,
                decision_tree_splitter, decision_tree_max_depth, decision_tree_min_sample_split,
                decision_tree_max_leaf_nodes)

    if select_train_model == 'RandomForestClassifier':
        select_training_model = st.sidebar.form(key='train_model')

        random_forest_criterion = select_training_model.selectbox('criterion', ['gini', 'entropy'])

        random_forest_n_estimators = select_training_model.selectbox('n_estimators', [i for i in range(1, 10000)])
        random_forest_max_depth = select_training_model.selectbox('max_depth', [None] + [i for i in range(1, 20)])
        random_forest_min_sample_split = select_training_model.selectbox('min_sample_split', [i for i in range(2, 20)])
        random_forest_max_leaf_nodes = select_training_model.selectbox('max_leaf_nodes',
                                                                       [None] + [i for i in range(2, 20)])
        random_forest_min_samples_leaf = select_training_model.selectbox('min_samples_leaf', [i for i in range(1, 20)])

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'RandomForestClassifier' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Classification_Final_Model.final_randomForestClassifier(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
                random_forest_criterion,
                random_forest_n_estimators, random_forest_max_depth, random_forest_min_sample_split,
                random_forest_max_leaf_nodes,
                random_forest_min_samples_leaf)

    if select_train_model == 'AdaBoostClassifier':
        select_training_model = st.sidebar.form(key='train_model')

        adaBoost_n_estimators = select_training_model.selectbox('n_estimators', [i for i in range(1, 10000)])
        adaBoost_learning_rate = select_training_model.selectbox('learning_rate', np.arange(0.1, 10, 0.01))
        adaBoost_algorithm = select_training_model.selectbox('algorithm', ['SAMME', 'SAMME.R'])

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'AdaBoostClassifier' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Classification_Final_Model.final_adaBoost(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
                adaBoost_n_estimators,
                adaBoost_learning_rate, adaBoost_algorithm)

    if select_train_model == 'GradientBoostingClassifier':
        select_training_model = st.sidebar.form(key='train_model')

        gbBoost_learing_rate = select_training_model.selectbox('learning_rate', np.arange(0.1, 10, 0.01))
        gbBoost_n_estimators = select_training_model.selectbox('n_estimators', [i for i in range(1, 10000)])
        gbBoost_max_depth = select_training_model.selectbox('max_depth', [None] + [i for i in range(1, 20)])
        gbBoost_min_sample_split = select_training_model.selectbox('min_sample_split', [i for i in range(2, 20)])
        gbBoost_max_leaf_nodes = select_training_model.selectbox('max_leaf_nodes', [None] + [i for i in range(2, 20)])
        gbBoost_min_samples_leaf = select_training_model.selectbox('min_samples_leaf', [i for i in range(1, 20)])

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'GradientBoostingClassifier' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Classification_Final_Model.final_gradientBoostClassifier(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
                gbBoost_learing_rate,
                gbBoost_n_estimators, gbBoost_max_depth, gbBoost_min_sample_split, gbBoost_max_leaf_nodes,
                gbBoost_min_samples_leaf)

    if select_train_model == 'XGBClassifier':
        select_training_model = st.sidebar.form(key='train_model')

        xgboost_learing_rate = select_training_model.selectbox('learning_rate', np.arange(0, 1, 0.01))
        xgboost_n_estimators = select_training_model.selectbox('n_estimators', [i for i in range(1, 10000)])
        xgboost_max_depth = select_training_model.selectbox('max_depth', [None] + [i for i in range(1, 20)])
        xgboost_min_child_weight = select_training_model.selectbox('min_child_weight', [i for i in range(1, 20)])
        xgboost_gamma = select_training_model.selectbox('gamma', np.arange(0, 1, 0.01))
        xgboost_subsample = select_training_model.selectbox('subsample', np.arange(0, 1, 0.01))
        xgboost_colsample_bytree = select_training_model.selectbox('colsample_bytree', np.arange(0, 1, 0.01))

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'XGBClassifier' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Classification_Final_Model.final_xgBClassifier(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
                xgboost_learing_rate,
                xgboost_n_estimators, xgboost_max_depth, xgboost_min_child_weight, xgboost_gamma, xgboost_subsample,
                xgboost_colsample_bytree)

    if select_train_model == 'LGBMModel':
        select_training_model = st.sidebar.form(key='train_model')

        lgbm_num_leaves = select_training_model.selectbox('num_leaves', [i for i in range(2, 100)])
        lgbm_n_estimators = select_training_model.selectbox('n_estimators', [i for i in range(1, 100000)])
        lgbm_max_depth = select_training_model.selectbox('max_depth', [None] + [i for i in range(1, 20)])
        lgbm_min_child_weight = select_training_model.selectbox('min_child_weight', [i for i in range(1, 20)])
        lgbm_learning_rate = select_training_model.selectbox('learning_rate', np.arange(0.1, 1, 0.01))

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'LGBMModel' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Classification_Final_Model.final_lgbmClassifier(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice, lgbm_num_leaves,
                lgbm_n_estimators, lgbm_max_depth, lgbm_min_child_weight, lgbm_learning_rate)

    if select_train_model == 'CatBoostClassifier':
        select_training_model = st.sidebar.form(key='train_model')

        catboost_iterations = select_training_model.selectbox('iterations', [i for i in range(1000, 10000)])
        catboost_learning_rate = select_training_model.selectbox('learning_rate', np.arange(0.1, 1, 0.001))
        catboost_depth = select_training_model.selectbox('depth', [i for i in range(1, 50)])
        catboost_l2_leaf_reg = select_training_model.selectbox('l2_leaf_reg', [i for i in range(1, 100)])

        st.session_state.train_model_button = select_training_model.form_submit_button('TRAIN MODEL')

        if select_train_model == 'CatBoostClassifier' and st.session_state.train_model_button:
            st.session_state.final_trained_model, st.session_state.final_training_error, st.session_state.final_testing_error = Classification_Final_Model.final_catBoostClassifier(
                st.session_state.final_df, st.session_state.y, st.session_state.error_function_choice,
                catboost_iterations,
                catboost_learning_rate, catboost_depth, catboost_l2_leaf_reg)

    if st.session_state.final_trained_model is not None:
        st.subheader('MODEL TRAINED : ')
        st.success(st.session_state.final_trained_model)

    return st.session_state.final_trained_model
