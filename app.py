import streamlit as st

# Set page title with icon
st.set_page_config(page_title='Horse Racing Predictor', page_icon='🐎')

from predict_page import show_predict_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("PUNTERS WELCOME!", ["Predict your horse now!", "Any tips to bet on horses?"])

if page == 'Predict your horse now!':
    show_predict_page()
else:
    show_explore_page()


