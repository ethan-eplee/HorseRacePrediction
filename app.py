import streamlit as st

from predict_page import show_predict_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("PUNTERS WELCOME!", ["", "Any tips to bet on horses?"])

if page == 'Predict your horse now!':
    show_predict_page()
else:
    show_explore_page()