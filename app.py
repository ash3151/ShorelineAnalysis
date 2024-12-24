import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

svr_models = []
folder_path = 'svr_models'

model_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
model_files = sorted(model_files, key=lambda x: int(x.split('_')[2].split('.')[0]))

Points=pd.read_csv("DF.csv")
for i in range(1984, 2023):
    c = Points[str(i)]
    c = c.ffill()
    Points[str(i)] = c

for model in model_files:
    if model.endswith('.pkl'):
        file_path = os.path.join(folder_path, model)
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
            svr_models.append(loaded_model)


def Prediction_svr(year):
    pred_points = []
    year = np.array([year]).reshape(-1, 1)
    for rbf_model in svr_models:
        prediction = rbf_model.predict(year)
        pred_points.append(prediction)
    return pred_points


def generate_scatter_plot(year):
    plt.figure(figsize=(10, 6))
    plt.scatter(Prediction_svr(year),Points["Axis"],s=0.5,color="Blue",label="Predicted Shoreline")
    # plt.scatter(Points["1984"],Points["Axis"],s=0.5,color="Grey",label='Observed Shoreline')
    plt.gca().invert_yaxis()
    plt.legend()
    return plt

st.title("Shoreline Plot Analysis")
st.write("Enter a year to generate the scatter plot.")
year_input = st.text_input("Year (e.g., 2023):", value="2023")

if st.button("Generate Plot"):
    try:
        year = int(year_input)
        scatter_plot = generate_scatter_plot(year)
        st.pyplot(scatter_plot)  
    except ValueError:
        st.error("Please enter a valid year.")