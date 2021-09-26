import streamlit as st
import pandas as pd
from Utils import Utility_Functions


def regression_prediction(df, target_column):
    if 'test_prediction' not in st.session_state:
        st.session_state.test_prediction = None

    df = st.session_state.preprocessing_Object.high_unique.transform(df)
    numerical_data, categorical_data = st.session_state.preprocessing_Object.split_numerical_categorical.split_numerical_categorical_transform(df)
    numerical_data = st.session_state.preprocessing_Object.numerical_imputer.transform(numerical_data)
    numerical_data= st.session_state.preprocessing_Object.correlation.transform(numerical_data)
    if st.session_state.outlier_choice:
        numerical_data = st.session_state.preprocessing_Object.outlier.transform(numerical_data)
    numerical_data = st.session_state.preprocessing_Object.standardize.transform(numerical_data)
    categorical_data = Utility_Functions.replace_other_nan_in_categorical_data(categorical_data)
    categorical_data = st.session_state.preprocessing_Object.categoric_imputer.transform(categorical_data)
    categorical_data = st.session_state.preprocessing_Object.annova.tranform(categorical_data)
    categorical_data = st.session_state.preprocessing_Object.encoder.transform(categorical_data)

    final_df = pd.concat([numerical_data, categorical_data], axis=1)

    st.session_state.test_prediction = st.session_state.final_trained_model.predict(final_df)

    st.session_state.prediction_result = pd.DataFrame({
        target_column: st.session_state.test_prediction
    })

    return st.session_state.prediction_result


def classification_prediction(df, target_column):
    if 'test_prediction' not in st.session_state:
        st.session_state.test_prediction = None

    df = st.session_state.preprocessing_Object.high_unique.transform(df)
    numerical_data, categorical_data = st.session_state.preprocessing_Object.split_numerical_categorical.split_numerical_categorical_transform(df)
    numerical_data = st.session_state.preprocessing_Object.numerical_imputer.transform(numerical_data)
    numerical_data = st.session_state.preprocessing_Object.annova.tranform(numerical_data)
    if st.session_state.outlier_choice:
        numerical_data = st.session_state.preprocessing_Object.outlier.transform(numerical_data)
    numerical_data = st.session_state.preprocessing_Object.standardize.transform(numerical_data)
    categorical_data = Utility_Functions.replace_other_nan_in_categorical_data(categorical_data)
    categorical_data = st.session_state.preprocessing_Object.categoric_imputer.transform(categorical_data)

    categorical_data = st.session_state.preprocessing_Object.encoder.transform(categorical_data)

    final_df = pd.concat([numerical_data, categorical_data], axis=1)

    st.session_state.test_prediction = st.session_state.final_trained_model.predict(final_df)

    st.session_state.prediction_result = pd.DataFrame({
        target_column: st.session_state.test_prediction
    })

    return st.session_state.prediction_result
