import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit import session_state as state
import os
import numpy as np
from PIL import Image


st.set_page_config(layout="wide")
df1 = pd.DataFrame(pd.read_csv('Ops_Session_Data.csv', encoding='latin1'))
df2 = pd.DataFrame(pd.read_csv('past_bookings_May23.csv', encoding='latin1'))

df1 = df1.dropna(subset=["uid"])
merged_df = pd.merge(df2, df1, on=["uid"])

requiredcols = ['Actual Date', 'EPOD Name', 'Customer Location City']
df = merged_df[requiredcols]


# Define the valid username and password
VALID_USERNAME = "admin"


def check_credentials():

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if username == VALID_USERNAME and password == st.secrets["password"]:
        st.session_state["logged_in"] = True
    else:
        st.warning("Invalid username or password.")


def main_page():
    st.title("Main Page")
    st.write("Welcome to the main page!")
    image = Image.open('Hpcharge.png')
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col3.image(image, use_column_width=False)
    with col1:
        df['Actual Date'] = pd.to_datetime(df['Actual Date'], errors='coerce')
        min_date = df['Actual Date'].min().date()
        max_date = df['Actual Date'].max().date()
        start_date = st.date_input(
            'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="epod-date-start")
    with col2:
        end_date = st.date_input(
            'End Date', min_value=min_date, max_value=max_date, value=max_date, key="epod-date-end")
    df['EPOD Name'] = df['EPOD Name'].str.replace('-', '')

    epods = df['EPOD Name'].unique()
    with col3:
        EPod = st.multiselect(label='Select The EPOD',
                              options=['All'] + epods.tolist(),
                              default='All')

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = df[(df['Actual Date'] >= start_date)
                       & (df['Actual Date'] <= end_date)]
    if 'All' in EPod:
        EPod = epods

    filtered_data = filtered_data[
        (filtered_data['EPOD Name'].isin(EPod))]


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_page()
else:
    check_credentials()
    if st.session_state.logged_in:
        st.experimental_rerun()
