import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

from Encoder.Hash_Encoder import Hash_Encode

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

train = pd.read_csv("/home/local/ZOHOCORP/prithivi-pt4242/Documents/ML_problems/titanic/train.csv")
test = pd.read_csv("/home/local/ZOHOCORP/prithivi-pt4242/Documents/ML_problems/titanic/test.csv")


# y = train['Survived']
# train.drop('Survived', inplace=True, axis=1)
#
# numerical_data = train.select_dtypes(['int', 'float'])
# categorical_data = train.select_dtypes(['object'])
#
# test_numerical_data = test.select_dtypes(['int', 'float'])
# test_categorical_data = test.select_dtypes(['object'])
#
# ob = Hash_Encode()
# categorical_data = ob.fit_transform(categorical_data)
# # op = ob.transform(categorical_data)
# print(categorical_data)
