import pandas as pd
import streamlit as st
import os

from reusable_components import get_logo, get_app_info
import plotly.express as px


def get_navigation_bar():
    st.sidebar.title("Navigation")
    get_app_info()

def get_page_content():
    project_title = st.markdown("<h1 style='text-align: center; color: white;'>CancerLncRNA</h1>",
                unsafe_allow_html=True)
    get_logo()
    hackathon_title = st.markdown("<h3 style='text-align: center; color: white;'>D4Gen Hacking Days</h3>",
                unsafe_allow_html=True)
    prediction_title = st.subheader("Cancer prediction")
    prediction_description = st.caption("Given a Gencode id, we predict the percentage to have a cancer.")
    input_id = st.text_input(label='Gencode id')
    run_button = st.button(label='Run analysis')
    if run_button:
        return input_id

def _load_outputs(gencode_id):
    """Load the outputs (cancers and functions) from gencode_id"""
    output = pd.read_csv(os.path.join('data', 'results', 'predictions.csv'))
    c_output = output[output.gencode_id == gencode_id].iloc[0, :][1:].astype(float)
    cancers = c_output[:30].sort_values(
        ascending=False)
    functions = c_output[-5:].sort_values(
        ascending=False)
    return cancers, functions

def get_results(gencode_id):
    # Inference of the model
    cancers, functions = _load_outputs(gencode_id)
    subtitle = st.subheader('Results')
    page_cols = st.columns((1, 1))
    page_cols[0].subheader('Cancers : ')
    page_cols[1].subheader('Functions: ')
    for i,(cancer, score) in enumerate(cancers[:3].iteritems()):
        page_cols[0].markdown(f"<h3 style='text-align: center; color: white;'>{i+1}. {cancer}</h3>",
                    unsafe_allow_html=True)
        page_cols[0].markdown(f"<h3 style='text-align: center; color: green;'>Score : {score}</h3>",
                              unsafe_allow_html=True)
    for i, (function, score) in enumerate(functions[:3].iteritems()):
        page_cols[1].markdown(
            f"<h3 style='text-align: center; color: white;'>{i + 1}. {function}</h3>",
            unsafe_allow_html=True)
        page_cols[1].markdown(
            f"<h3 style='text-align: center; color: blue;'>Score : {score}</h3>",
            unsafe_allow_html=True)
    return cancers, functions


def visualize_outputs(cancers, functions) :
    """Visualize the outputs."""
    st.subheader("Cancers")
    fig = px.bar(cancers[:10], orientation='h')
    fig.update_layout(
        xaxis_title="Score",
        yaxis_title="Cancer",
    )
    st.write(fig)
    st.subheader("Functions")
    fig = px.bar(functions, orientation='h')
    fig.update_layout(
        xaxis_title="Score",
        yaxis_title="Cancer",
    )
    st.write(fig)


def main():
    """Main function"""
    st.set_page_config(page_title="TITLE", page_icon=":medical_symbol:", layout="wide")

    # Navigation bar
    get_navigation_bar()

    # Input
    gencode_id = get_page_content()

    # Results
    if gencode_id:
        cancers, functions = get_results(gencode_id)
        visualize_outputs(cancers, functions)

if __name__ == "__main__":
    main()
