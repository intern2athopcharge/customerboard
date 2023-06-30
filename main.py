import streamlit as st
import pandas as pd
from streamlit import session_state as state
import os

df1 = pd.DataFrame(pd.read_csv('Ops_Session_Data.csv', encoding='latin1'))
df2 = pd.DataFrame(pd.read_csv('past_bookings_May23.csv', encoding='latin1'))

df1 = df1.dropna(subset=["uid"])
merged_df = pd.merge(df2, df1, on=["uid"])

requiredcols = ['Actual Date', 'EPOD Name', 'Customer Location City']
merged_df = merged_df[requiredcols]


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


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_page()
else:
    check_credentials()
    if st.session_state.logged_in:
        st.experimental_rerun()
