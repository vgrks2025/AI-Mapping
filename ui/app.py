# AI Data Mapper Application - Faster Asset
# Input the database - source and target
# Analyse, Filter, Group/Domain Classification, Generation, Mapping
# Author: Yesh

# libraries
import streamlit as st
import streamlit.components.v1 as components



st.set_page_config(page_title="AI Data Mapper Application - Faster Asset", layout="wide")
st.logo("../data/logo.png",size="large")

st.title("AI Data Mapper Application")
st.subheader("Welcome! Please navigate following - ")
st.write("/ai - To generate AI Mappings")


api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

st.sidebar.write("Enter Database credentials")
ip = st.sidebar.text_input("Enter ip", type="password")
user = st.sidebar.text_input("Enter user", type="password")
password = st.sidebar.text_input("Enter password", type="password")
port = st.sidebar.text_input("Enter port", type="password")

if st.sidebar.button("Submit"):
    st.sidebar.write("All credentials submitted!")
    config = {
        "api_key":api_key,
        "ip": ip,
        "user": user,
        "password":password,
        "port": port
    }
    st.session_state['config'] = config 
