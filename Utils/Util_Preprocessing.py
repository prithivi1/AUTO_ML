import warnings

warnings.filterwarnings("ignore")

import streamlit as st
from Stage_One_Preprocessing.Numerical_Categorical_Split import Numerical_Categorical_Split
from Imputer.Simple_Imputer import Simple_Imputer
from Imputer.Mice_Imputer import Mice_Imputer
from Imputer.Manual_Imputer_Numeric import Manual_Numeric
from Feature_Extraction.Correlation import Correlation
from Outlier.IQR import IQR_Treatement
from Standardization.Standard_Scaler import Standard_Scaler
from Standardization.MinMaxScaler import Min_Max_Scaler
from Imputer.Mode_Imputer import Mode_Imputer
from Imputer.Manual_Imputer import Manual_Imputer
from Imputer.Constant_imputer import Constant_Imputer
from Feature_Extraction.Annova import Annova
from Encoder.One_Hot_Encoding import One_Hot
from Encoder.Label_Encode import Label_Encode
from Encoder.Frequency_Encoder import Frequency_Encoding
from Encoder.Target_Encoder import Target_Encode
from Encoder.BaseN_Encoder import BaseN_Encode
from Encoder.Hash_Encoder import Hash_Encode
from Stage_One_Preprocessing.Remove_Unique import Remove_Unique


class Preprocess:

    def __init__(self):
        self.split_numerical_categorical = Numerical_Categorical_Split()
        self.numerical_imputer = None
        self.correlation = None
        self.outlier = None
        self.standardize = None
        self.categoric_imputer = None
        self.annova = None
        self.encoder = None
        self.high_unique = None

    def numerical_categorical_split(self, data):
        self.split_numerical_categorical = Numerical_Categorical_Split()
        numerical_data, categorical_data = self.split_numerical_categorical.split_numerical_categorical_fit_transform(
            data.copy())
        return numerical_data, categorical_data

    def select_best_numerical_imputation(self, numerical_data, numerical_missing_imputation='Manual'):
        if numerical_missing_imputation == 'Manual':
            self.numerical_imputer = Manual_Numeric()
            numerical_data = self.numerical_imputer.fit_transform(numerical_data)
        elif numerical_missing_imputation == 'Mice_Imputation':
            self.numerical_imputer = Mice_Imputer()
            numerical_data = self.numerical_imputer.fit_transform(numerical_data)
        else:
            self.numerical_imputer = Simple_Imputer()
            numerical_data = self.numerical_imputer.fit_transform(numerical_data, numerical_missing_imputation)
        return numerical_data

    def correlation_testing(self, numerical_data, y):
        self.correlation = Correlation()
        numerical_data, corr_data, columns_to_drop = self.correlation.fit_transform(numerical_data, y)
        return numerical_data, corr_data, columns_to_drop

    def select_best_Outlier_Treatement(self, numerical_data, oulier_choice):
        if oulier_choice:
            self.outlier = IQR_Treatement()
            numerical_data, continuous_column = self.outlier.fit_transform(numerical_data)
            return numerical_data, continuous_column
        else:
            return numerical_data, list()

    def select_best_standardization(self, numerical_data, standardising='Standard_Scalar'):
        if standardising == 'Standard_Scalar':
            self.standardize = Standard_Scaler()
            numerical_data = self.standardize.fit_transform(numerical_data)
        elif standardising == 'MinMaxScalar':
            self.standardize = Min_Max_Scaler()
            numerical_data = self.standardize.fit_transform(numerical_data)

        return numerical_data

    def select_best_categorical_imputation(self, categorical_data, categorical_missing_imputation='Manual'):
        if categorical_missing_imputation == 'Manual':
            self.categoric_imputer = Manual_Imputer()
            categorical_data = self.categoric_imputer.fit_transform(categorical_data)
        elif categorical_missing_imputation == 'Mode':
            self.categoric_imputer = Mode_Imputer()
            categorical_data = self.categoric_imputer.fit_transform(categorical_data)
        elif categorical_missing_imputation == 'Constant':
            self.categoric_imputer = Constant_Imputer()
            categorical_data = self.categoric_imputer.fit_transform(categorical_data)

        return categorical_data

    def annova_testing(self, categorical_data, y):
        self.annova = Annova()
        data, annova_data, dropped_cols = self.annova.fit_transform(categorical_data, y)
        return data, annova_data, dropped_cols

    def select_best_encoding_technique(self, categorical_data, encoding='Label_Encoding'):
        if encoding == 'Label_Encoding':
            self.encoder = Label_Encode()
            categorical_data = self.encoder.fit_transform(categorical_data)
        elif encoding == 'One_Hot_Encoding':
            self.encoder = One_Hot()
            categorical_data = self.encoder.transform(categorical_data)
        elif encoding == 'Frequency_Encoding':
            self.encoder = Frequency_Encoding()
            categorical_data = self.encoder.fit_transform(categorical_data)
        elif encoding == 'Target_Encoding':
            self.encoder = Target_Encode()
            categorical_data = self.encoder.fit_transform(categorical_data, st.session_state.y)
        elif encoding == 'BaseN_Encoding':
            self.encoder = BaseN_Encode()
            categorical_data = self.encoder.fit_transform(categorical_data)
        elif encoding == 'Hash_Encoding':
            self.encoder = Hash_Encode()
            categorical_data = self.encoder.fit_transform(categorical_data)
        return categorical_data

    def remove_high_unique(self, data):
        self.high_unique = Remove_Unique()
        data = self.high_unique.fit_transform(data)
        return data
