import streamlit as st
import pickle
from Utils.Utility_Functions import *
from Utils.Util_Preprocessing import *


def regression_Preprocessing():

    if 'numerical_imputer_choice' not in st.session_state:
        st.session_state.numerical_imputer_choice = None

    if 'outlier_choice' not in st.session_state:
        st.session_state.outlier_choice = None

    if 'standardiser_choice' not in st.session_state:
        st.session_state.standardiser_choice = None

    if 'categorical_imputor_choice' not in st.session_state:
        st.session_state.categorical_imputor_choice = None

    if 'encoder_choice' not in st.session_state:
        st.session_state.encoder_choice = None

    if 'numerical_data' not in st.session_state:
        st.session_state.numerical_data = None

    if 'categorical_data' not in st.session_state:
        st.session_state.categorical_data = None

    if 'numerical_imputed_data' not in st.session_state:
        st.session_state.numerical_imputed_data = None

    if 'numerical_corr_data' not in st.session_state:
        st.session_state.numerical_corr_data = None

    if 'corr_data' not in st.session_state:
        st.session_state.corr_data = None

    if 'corr_dropped_cols' not in st.session_state:
        st.session_state.corr_dropped_cols = None

    if 'numerical_outlier_data' not in st.session_state:
        st.session_state.numerical_outlier_data = None

    if 'continuous_column' not in st.session_state:
        st.session_state.continuous_column = None

    if 'numerical_standarized_data' not in st.session_state:
        st.session_state.numerical_standarized_data = None

    if 'categorical_imputed_data' not in st.session_state:
        st.session_state.categorical_imputed_data = None

    if 'categorical_annova_data' not in st.session_state:
        st.session_state.categorical_annova_data = None

    if 'annova_test' not in st.session_state:
        st.session_state.annova_test = None

    if 'annova_dropped_cols' not in st.session_state:
        st.session_state.annova_dropped_cols = None

    if 'categorical_encoded_data' not in st.session_state:
        st.session_state.categorical_encoded_data = None

    if 'unique_dropped_cols' not in st.session_state:
        st.session_state.unique_dropped_cols = None

    if 'regression' not in st.session_state:
        st.session_state.regression = None

    if st.session_state.regression is None:
        st.session_state.regression = Preprocess()

    regressiong_preprocessing = st.sidebar.form(key='regressiong_preprocessing')

    regressiong_preprocessing.header('PREPROCESSING CONFIG')

    st.session_state.numerical_imputer_choice = regressiong_preprocessing.selectbox('Select Numerical Imputation '
                                                                                    'Type:', ['Manual', 'mean',
                                                                                              'median',
                                                                                              'most_frequent',
                                                                                              'constant',
                                                                                              'Mice_Imputation'])

    st.session_state.outlier_choice = regressiong_preprocessing.checkbox('Treat_Outlier', value=True)

    st.session_state.standardiser_choice = regressiong_preprocessing.selectbox('Select best Standardization:',
                                                                               ['Standard_Scalar', 'MinMaxScalar'])

    st.session_state.categorical_imputor_choice = regressiong_preprocessing.selectbox('Select Categorical '
                                                                                      'Imputation Type:',
                                                                                      ['Manual',
                                                                                       'Mode', 'Constant'])

    st.session_state.encoder_choice = regressiong_preprocessing.selectbox('Select Encoding Technique:',
                                                                          ['Frequency_Encoding', 'Label_Encoding',
                                                                           'One_Hot_Encoding', 'Target_Encoding',
                                                                           'BaseN_Encoding', 'Hash_Encoding'])

    st.session_state.preprocess_button = regressiong_preprocessing.form_submit_button("PROCESS DATA")

    if st.session_state.preprocess_button:
        st.session_state.X, st.session_state.unique_dropped_cols = st.session_state.regression.remove_high_unique(
            st.session_state.X.copy())
        st.session_state.numerical_data, st.session_state.categorical_data = st.session_state.regression.numerical_categorical_split(
            st.session_state.X.copy())
        st.session_state.numerical_imputed_data = st.session_state.regression.select_best_numerical_imputation(
            st.session_state.numerical_data.copy(), st.session_state.numerical_imputer_choice)
        st.session_state.numerical_corr_data, st.session_state.corr_data, st.session_state.corr_dropped_cols = st.session_state.regression.correlation_testing(
            st.session_state.numerical_imputed_data.copy(), st.session_state.y.copy())
        st.session_state.numerical_outlier_data, st.session_state.continuous_column = st.session_state.regression.select_best_Outlier_Treatement(
            st.session_state.numerical_corr_data.copy(), st.session_state.outlier_choice)
        st.session_state.numerical_standarized_data = st.session_state.regression.select_best_standardization(
            st.session_state.numerical_outlier_data.copy(), st.session_state.standardiser_choice)
        st.session_state.categorical_data = replace_other_nan_in_categorical_data(
            st.session_state.categorical_data.copy())
        st.session_state.categorical_imputed_data = st.session_state.regression.select_best_categorical_imputation(
            st.session_state.categorical_data.copy(), st.session_state.categorical_imputor_choice)
        st.session_state.categorical_annova_data, st.session_state.annova_test, st.session_state.annova_dropped_cols = st.session_state.regression.annova_testing(
            st.session_state.categorical_imputed_data.copy(), st.session_state.y.copy())
        st.session_state.categorical_encoded_data = st.session_state.regression.select_best_encoding_technique(
            st.session_state.categorical_annova_data.copy(), st.session_state.encoder_choice)
        st.session_state.final_df = pd.concat(
            [st.session_state.numerical_standarized_data.copy(), st.session_state.categorical_encoded_data.copy()],
            axis=1)

    if st.session_state.numerical_data is not None:
        st.success('Numerical, Categorical Data Split Successfully')
        st.subheader('NUMERICAL DATA PROCESSING')
        st.write(st.session_state.numerical_data)

    if st.session_state.unique_dropped_cols is not None:
        st.subheader('DROPED UNIQUE DATA')
        st.write(st.session_state.unique_dropped_cols)

    if st.session_state.numerical_imputed_data is not None:
        st.subheader('IMPUTED NUMERICAL DATA')
        st.write(st.session_state.numerical_imputed_data)

    if st.session_state.corr_data is not None:
        st.subheader('CORRELATION DATA')
        st.write(st.session_state.corr_data)
        st.subheader('DROPPED COLUMNS')
        st.write(st.session_state.corr_dropped_cols)

    if st.session_state.numerical_outlier_data is not None:
        st.subheader('NUMERICAL OUTLIER TREATED COLUMNS')
        st.write(st.session_state.continuous_column)

    if st.session_state.numerical_standarized_data is not None:
        st.subheader('NUMERICAL STANDARDIZED DATA')
        st.write(st.session_state.numerical_standarized_data)

    if st.session_state.categorical_data is not None:
        st.subheader('CATEGORICAL DATA')
        st.write(st.session_state.categorical_data)

    if st.session_state.categorical_imputed_data is not None:
        st.subheader('IMPUTED CATEGORICAL DATA')
        st.write(st.session_state.categorical_imputed_data)

    if st.session_state.categorical_annova_data is not None:
        st.subheader('ANNOVA TEST')
        st.write(st.session_state.annova_test)
        st.subheader('DROPPED COLUMNS')
        st.write(st.session_state.annova_dropped_cols)

    if st.session_state.categorical_encoded_data is not None:
        st.subheader('ENCODED CATEGORICAL DATA')
        st.write(st.session_state.categorical_encoded_data)

    if st.session_state.final_df is not None:
        st.subheader('PREPROCESSED DATA')
        st.write(st.session_state.final_df)

    return st.session_state.final_df,st.session_state.regression


def classification_Preprocessing():

    if 'numerical_imputer_choice' not in st.session_state:
        st.session_state.numerical_imputer_choice = None

    if 'outlier_choice' not in st.session_state:
        st.session_state.outlier_choice = None

    if 'standardiser_choice' not in st.session_state:
        st.session_state.standardiser_choice = None

    if 'categorical_imputor_choice' not in st.session_state:
        st.session_state.categorical_imputor_choice = None

    if 'encoder_choice' not in st.session_state:
        st.session_state.encoder_choice = None

    if 'numerical_data' not in st.session_state:
        st.session_state.numerical_data = None

    if 'categorical_data' not in st.session_state:
        st.session_state.categorical_data = None

    if 'numerical_imputed_data' not in st.session_state:
        st.session_state.numerical_imputed_data = None

    if 'categorical_annova_data' not in st.session_state:
        st.session_state.categorical_annova_data = None

    if 'annova_test' not in st.session_state:
        st.session_state.annova_test = None

    if 'annova_dropped_cols' not in st.session_state:
        st.session_state.annova_dropped_cols = None

    if 'numerical_outlier_data' not in st.session_state:
        st.session_state.numerical_outlier_data = None

    if 'continuous_column' not in st.session_state:
        st.session_state.continuous_column = None

    if 'numerical_standarized_data' not in st.session_state:
        st.session_state.numerical_standarized_data = None

    if 'categorical_imputed_data' not in st.session_state:
        st.session_state.categorical_imputed_data = None

    if 'categorical_encoded_data' not in st.session_state:
        st.session_state.categorical_encoded_data = None

    if 'unique_dropped_cols' not in st.session_state:
        st.session_state.st.session_state.unique_dropped_cols = None

    if 'classification' not in st.session_state:
        st.session_state.classification = None

    if st.session_state.classification is None:
        st.session_state.classification = Preprocess()

    classification_preprocessing = st.sidebar.form(key='classification_preprocessing')

    classification_preprocessing.header('PREPROCESSING CONFIG')

    st.session_state.numerical_imputer_choice = classification_preprocessing.selectbox(
        'Select Numerical Imputation Type:',
        ['Manual', 'mean', 'median',
         'most_frequent',
         'constant',
         'Mice_Imputation'])

    st.session_state.outlier_choice = classification_preprocessing.checkbox('Treat_Outlier', value=True)

    st.session_state.standardiser_choice = classification_preprocessing.selectbox('Select best Standardization:',
                                                                                  ['Standard_Scalar', 'MinMaxScalar'])
    st.session_state.categorical_imputor_choice = classification_preprocessing.selectbox(
        'Select Categorical Imputation Type:',
        ['Manual', 'Mode', 'Constant'])

    st.session_state.encoder_choice = classification_preprocessing.selectbox('Select Encoding Technique:',
                                                                             ['Frequency_Encoding', 'Label_Encoding',
                                                                              'One_Hot_Encoding', 'Target_Encoding',
                                                                              'BaseN_Encoding', 'Hash_Encoding'])

    st.session_state.preprocess_button = classification_preprocessing.form_submit_button("PROCESS DATA")

    if st.session_state.preprocess_button:
        st.session_state.X, st.session_state.unique_dropped_cols = st.session_state.classification.remove_high_unique(
            st.session_state.X.copy())
        st.session_state.numerical_data, st.session_state.categorical_data = st.session_state.classification.numerical_categorical_split(
            st.session_state.X.copy())
        st.session_state.numerical_imputed_data = st.session_state.classification.select_best_numerical_imputation(
            st.session_state.numerical_data.copy(), st.session_state.numerical_imputer_choice)
        st.session_state.categorical_annova_data, st.session_state.annova_test, st.session_state.annova_dropped_cols = st.session_state.classification.annova_testing(
            st.session_state.numerical_imputed_data.copy(), st.session_state.y.copy())
        st.session_state.numerical_outlier_data, st.session_state.continuous_column = st.session_state.classification.select_best_Outlier_Treatement(
            st.session_state.categorical_annova_data.copy(), st.session_state.outlier_choice)
        st.session_state.numerical_standarized_data = st.session_state.classification.select_best_standardization(
            st.session_state.numerical_outlier_data.copy(), st.session_state.standardiser_choice)
        st.session_state.categorical_data = replace_other_nan_in_categorical_data(
            st.session_state.categorical_data.copy())
        st.session_state.categorical_imputed_data = st.session_state.classification.select_best_categorical_imputation(
            st.session_state.categorical_data.copy(), st.session_state.categorical_imputor_choice)
        st.session_state.categorical_encoded_data = st.session_state.classification.select_best_encoding_technique(
            st.session_state.categorical_imputed_data.copy(), st.session_state.encoder_choice)
        st.session_state.final_df = pd.concat(
            [st.session_state.numerical_standarized_data.copy(), st.session_state.categorical_encoded_data.copy()],
            axis=1)

    if st.session_state.numerical_data is not None:
        st.success('Numerical, Categorical Data Split Successfully')
        st.subheader('NUMERICAL DATA PROCESSING')
        st.write(st.session_state.numerical_data)

    if st.session_state.unique_dropped_cols is not None:
        st.subheader('DROPED UNIQUE DATA')
        st.write(st.session_state.unique_dropped_cols)

    if st.session_state.numerical_imputed_data is not None:
        st.subheader('IMPUTED NUMERICAL DATA')
        st.write(st.session_state.numerical_imputed_data)

    if st.session_state.categorical_annova_data is not None:
        st.subheader('ANNOVA TEST')
        st.write(st.session_state.annova_test)
        st.subheader('DROPPED COLUMNS')
        st.write(st.session_state.annova_dropped_cols)

    if st.session_state.numerical_outlier_data is not None:
        st.subheader('NUMERICAL OUTLIER TREATED COLUMNS')
        st.write(st.session_state.continuous_column)

    if st.session_state.numerical_standarized_data is not None:
        st.subheader('NUMERICAL STANDARDIZED DATA')
        st.write(st.session_state.numerical_standarized_data)

    if st.session_state.categorical_data is not None:
        st.subheader('CATEGORICAL DATA')
        st.write(st.session_state.categorical_data)

    if st.session_state.categorical_imputed_data is not None:
        st.subheader('IMPUTED CATEGORICAL DATA')
        st.write(st.session_state.categorical_imputed_data)

    if st.session_state.categorical_encoded_data is not None:
        st.subheader('ENCODED CATEGORICAL DATA')
        st.write(st.session_state.categorical_encoded_data)

    if st.session_state.final_df is not None:
        st.subheader('PREPROCESSED DATA')
        st.write(st.session_state.final_df)

    return st.session_state.final_df,st.session_state.classification
