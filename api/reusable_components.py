import streamlit as st
import cv2
import os


def get_logo():
    img = cv2.imread(os.path.join('api', 'images', 'logo.png'))
    col1, col2, col3 = st.columns([0.2, 0.1, 0.2])
    col2.image(img, use_column_width=True)


def get_app_info():
    st.sidebar.info(
        """
         The present project consists on the development of an AI method based on deep-learning to associate long non-coding RNAs (lncRNAs) to cancer types and biological functions.
        """)
    return
