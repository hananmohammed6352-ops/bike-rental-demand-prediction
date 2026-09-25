import streamlit as st
import gzip
import pickle

st.title("Bike Rental Demand Prediction")

with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.success("Model loaded successfully!")

st.write("Number of features:", len(feature_columns))
