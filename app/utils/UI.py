from PIL import Image
import streamlit as st

def add_logo(path,slogan):
    logo = Image.open(path)
    modified_logo = logo.resize((100, 100))
    col1, col2 = st.sidebar.columns([1,6])
    col1.image(modified_logo)
    col2.header(slogan)