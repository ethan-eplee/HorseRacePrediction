import streamlit as st

from predict_page import show_predict_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("PUNTERS WELCOME!", ["Any tips to bet on horses?", "Predict your horse now!"])

if page == 'Any tips to bet on horses?':
    show_explore_page()
else:
    show_predict_page()