import streamlit as st
from Utils import Utility_Functions

st.set_option('deprecation.showPyplotGlobalUse', False)


def regression_base_prediction():
    if 'st.session_state.error_function_choice' not in st.session_state:
        st.session_state.error_function_choice = None

    if 'st.session_state.n_split_choice' not in st.session_state:
        st.session_state.n_split_choice = None

    if 'model_building' not in st.session_state:
        st.session_state.model_building = None

    regression_model_Building = st.sidebar.form(key='regression_model_Building')

    regression_model_Building.header('MODEL BUILDING CONFIG')

    st.session_state.error_function_choice = regression_model_Building.selectbox('Select Error Metrics:',
                                                                                 ['neg_mean_absolute_error',
                                                                                  'explained_variance', 'max_error',
                                                                                  'neg_mean_squared_error',
                                                                                  'neg_root_mean_squared_error',
                                                                                  'neg_mean_squared_log_error',
                                                                                  'neg_median_absolute_error',
                                                                                  'r2',
                                                                                  'neg_mean_poisson_deviance',
                                                                                  'neg_mean_gamma_deviance',
                                                                                  'neg_mean_absolute_percentage_error'])

    st.session_state.n_split_choice = regression_model_Building.slider("Select K-fold", 2, 10)

    st.session_state.model_building = regression_model_Building.form_submit_button("Model_Build")

    if st.session_state.model_building:
        st.session_state.base_models = Utility_Functions.run_models(st.session_state.final_df.copy(),
                                                                    st.session_state.y.copy(),
                                                                    st.session_state.error_function_choice,
                                                                    st.session_state.n_split_choice,
                                                                    'Regression')

    if st.session_state.base_models is not None:
        st.subheader('MODEL PERFORMANCE')

        for i in st.session_state.base_models:
            st.subheader(i)
            st.write((st.session_state.base_models[i]['Error']).astype(str))
            st.success('TRAINING AVG MEAN : ' + str(
                sum(st.session_state.base_models[i]['Error']['Training Error']) / len(
                    st.session_state.base_models[i]['Error']['Training Error'])))
            st.success('TESTING AVG ERROR : ' + str(
                sum(st.session_state.base_models[i]['Error']['Testing Error']) / len(
                    st.session_state.base_models[i]['Error']['Testing Error'])))
            st.session_state.base_models[i]['Error'].plot(x='Trail', y=['Training Error', 'Testing Error'])
            st.pyplot()

    return st.session_state.base_models


def classification_base_prediction():
    if 'st.session_state.error_function_choice' not in st.session_state:
        st.session_state.error_function_choice = None

    if 'st.session_state.n_split_choice' not in st.session_state:
        st.session_state.n_split_choice = None

    if 'model_building' not in st.session_state:
        st.session_state.model_building = None

    classification_model_Building = st.sidebar.form(key='classification_model_Building')

    classification_model_Building.header('MODEL BUILDING CONFIG')

    st.session_state.error_function_choice = classification_model_Building.selectbox('Select Error Metrics:',
                                                                                     ['accuracy', 'balanced_accuracy',
                                                                                      'top_k_accuracy',
                                                                                      'average_precision',
                                                                                      'neg_brier_score', 'f1',
                                                                                      'neg_log_loss',
                                                                                      'precision'
                                                                                      'recall', 'jaccard', 'roc_auc',
                                                                                      'roc_auc_ovr', 'roc_auc_ovo',
                                                                                      'roc_auc_ovr_weighted',
                                                                                      'roc_auc_ovo_weighted'])

    st.session_state.n_split_choice = classification_model_Building.slider("Select K-fold", 2, 10)

    st.session_state.model_building = classification_model_Building.form_submit_button("Model_Build")

    if st.session_state.model_building:
        st.session_state.base_models = Utility_Functions.run_models(st.session_state.final_df.copy(),
                                                                    st.session_state.y.copy(),
                                                                    st.session_state.error_function_choice,
                                                                    st.session_state.n_split_choice,
                                                                    'Classification')

    if st.session_state.base_models is not None:
        st.subheader('MODEL PERFORMANCE')

        for i in st.session_state.base_models:
            st.subheader(i)
            st.write((st.session_state.base_models[i]['Error']).astype(str))
            st.success('AVG TRAINING ACCURACY : ' + str(
                sum(st.session_state.base_models[i]['Error']['Training Accuracy']) / len(
                    st.session_state.base_models[i]['Error']['Training Accuracy'])))
            st.success('AVG TESTING ACCURACY : ' + str(
                sum(st.session_state.base_models[i]['Error']['Testing Accuracy']) / len(
                    st.session_state.base_models[i]['Error']['Testing Accuracy'])))
            st.session_state.base_models[i]['Error'].plot(x='Trail', y=['Training Accuracy', 'Testing Accuracy'])
            st.pyplot()

    return st.session_state.base_models
