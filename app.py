import streamlit as st
import pandas as pd
import joblib
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')
