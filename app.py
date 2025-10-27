import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

countries = ['India', 'United States', 'China', 'Germany']

def predict_emissions(country):
    model = joblib.load(f"models/{country}_model.pkl")
    years = np.arange(2024, 2034).reshape(-1, 1)
    predictions = model.predict(years)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(years, predictions, 'g--', label='Predicted Emissions')
    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂ Emissions (mt)")
    ax.set_title(f"Projected CO₂ Emissions: {country}")
    ax.legend()
    ax.grid(True)
    
    return fig

demo = gr.Interface(
    fn=predict_emissions,
    inputs=gr.Dropdown(choices=countries, label="Select Country"),
    outputs=gr.Plot(label="Emission Projection"),
    title="🌍 AI-Powered CO₂ Emission Predictor",
    description="Predict and visualize future CO₂ emissions for selected countries using ML models."
)

demo.launch()
