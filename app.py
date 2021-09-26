import streamlit as st
import pandas as pd

import app_HyperTuning
import app_Model_Building
import app_Prediction
import app_preprocessing
import app_Base_Prediction
import app_graph

if 'df' not in st.session_state:
    st.session_state.df = None

if 'plot1' not in st.session_state:
    st.session_state.plot1 = None

if 'plot2' not in st.session_state:
    st.session_state.plot2 = None

if 'X' not in st.session_state:
    st.session_state.X = None

if 'y' not in st.session_state:
    st.session_state.y = None

if 'final_df' not in st.session_state:
    st.session_state.final_df = None

if 'preprocessing_Object' not in st.session_state:
    st.session_state.preprocessing_Object = None

if 'base_models' not in st.session_state:
    st.session_state.base_models = None

if 'final_trained_model' not in st.session_state:
    st.session_state.final_trained_model = None

if 'final_training_error' not in st.session_state:
    st.session_state.final_training_error = None

if 'final_testing_error' not in st.session_state:
    st.session_state.final_testing_error = None

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

st.title('AUTO ML')

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    st.header('DISPLAYING DATASET')
    st.session_state.df = pd.read_csv(uploaded_file)
    st.write(st.session_state.df)

    st.sidebar.header('SELECT HERE')
    lst = st.session_state.df.columns
    target_column = st.sidebar.selectbox('Select Target column:', lst)
    st.session_state.problem_type = st.sidebar.radio("Select : ", ('REGRESSION', 'CLASSIFICATION'))

    graph_plot = st.sidebar.form(key='graph_plot')

    graph_plot.header('PLOTS')

    plot1 = graph_plot.selectbox('Select X column:', options=[None]+list(lst))
    plot2 = graph_plot.selectbox('Select Y column:', [None]+list(lst))

    plot_type = graph_plot.selectbox('PLOT TYPE', ['BOX PLOT', 'BAR PLOT', 'HIST PLOT', 'SCATTER PLOT', 'VIOLIN PLOT', 'COUNT PLOT'])

    pairPlot = graph_plot.checkbox('PAIR PLOT', value=True)
    disPlot = graph_plot.checkbox('DISPLOT', value=True)

    st.session_state.plot_graph = graph_plot.form_submit_button("PLOT GRAPH")

    if st.session_state.plot_graph:
        app_graph.plot_graph(plot1, plot2, plot_type)
        if pairPlot:
            app_graph.pairPlot()
        if disPlot:
            app_graph.disPlot()

    st.session_state.y = st.session_state.df[target_column]
    st.session_state.X = st.session_state.df.drop(target_column, axis=1)

    if st.session_state.X is not None:
        st.subheader('INDEPENDENT DATA')
        st.write(st.session_state.X)
        st.subheader('DEPENDENT DATA')
        st.write(st.session_state.y)

    if st.session_state.X is not None and st.session_state.problem_type and (
            st.session_state.problem_type == 'REGRESSION'):
        st.session_state.final_df,st.session_state.preprocessing_Object = app_preprocessing.regression_Preprocessing()

    # st.sidebar.selectbox('help',['a','b','c'])

    if st.session_state.X is not None and st.session_state.problem_type and (
            st.session_state.problem_type == 'CLASSIFICATION'):
        st.session_state.final_df,st.session_state.preprocessing_Object = app_preprocessing.classification_Preprocessing()

    if st.session_state.final_df is not None and (st.session_state.problem_type == 'REGRESSION'):
        st.session_state.base_models = app_Base_Prediction.regression_base_prediction()

    if st.session_state.final_df is not None and (st.session_state.problem_type == 'CLASSIFICATION'):
        st.session_state.base_models = app_Base_Prediction.classification_base_prediction()

    if st.session_state.base_models is not None and (st.session_state.problem_type=='REGRESSION'):
        app_HyperTuning.regression_hypertuning()

    if st.session_state.base_models is not None and (st.session_state.problem_type=='CLASSIFICATION'):
        app_HyperTuning.classification_hypertuning()

    if st.session_state.base_models is not None and (st.session_state.problem_type=='REGRESSION'):
        st.session_state.final_trained_model = app_Model_Building.regression_model_building()

    if st.session_state.base_models is not None and (st.session_state.problem_type=='CLASSIFICATION'):
        st.session_state.final_trained_model = app_Model_Building.classification_model_building()


if st.session_state.final_trained_model is not None:

    testing_data_form = st.sidebar.form(key='testing_data')

    testing_file = testing_data_form.file_uploader("Choose a Test file")

    if testing_file is not None:
        test_df = pd.read_csv(testing_file)
        st.subheader('TESTING DATA')
        st.write(test_df)

    predict_button = testing_data_form.form_submit_button('PREDICT')

    if predict_button and st.session_state.problem_type == 'REGRESSION':
        st.session_state.prediction_result = app_Prediction.regression_prediction(test_df, target_column)

    if predict_button and st.session_state.problem_type == 'CLASSIFICATION':
        st.session_state.prediction_result = app_Prediction.classification_prediction(test_df, target_column)

    if st.session_state.prediction_result is not None:
        st.subheader('YOUR PREDICTION ')
        st.write(st.session_state.prediction_result)
        st.download_button(label='DOWNLOAD  .CSV',data=st.session_state.prediction_result.to_csv(index=False),mime='text/csv',file_name='Prediction.csv')
