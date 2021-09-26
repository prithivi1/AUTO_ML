import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def pairPlot():
    fig = sns.pairplot(st.session_state.df)
    st.pyplot(fig)


def disPlot():
    fig = sns.displot(st.session_state.df)
    st.pyplot(fig)


def plot_graph(x, y, type):

    if type == 'BOX PLOT':
        st.subheader('BOX PLOT')
        sns.boxplot(x=st.session_state.df[x], y=st.session_state.df[y])
        st.pyplot()

    elif type == 'BAR PLOT':
        st.subheader('BAR PLOT')
        sns.barplot(x=x, y=y, data=st.session_state.df)
        st.pyplot()

    elif type == 'HIST PLOT':
        st.subheader('HIST PLOT')
        sns.histplot(x=x, y=y, data=st.session_state.df)
        st.pyplot()

    elif type == 'SCATTER PLOT':
        st.subheader('SCATTER PLOT')
        sns.scatterplot(st.session_state.df[x], st.session_state.df[y])
        st.pyplot()

    elif type == 'VIOLIN PLOT':
        st.subheader('VIOLIN PLOT')
        sns.violinplot(x=st.session_state.df[x], y=st.session_state.df[y])
        st.pyplot()

    elif type == 'COUNT PLOT':
        st.subheader('COUNT PLOT')
        sns.countplot(st.session_state.df[x])
        st.pyplot()
