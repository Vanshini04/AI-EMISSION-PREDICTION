import pandas as pd
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# --- Paths ---
DATA_ZIP = "dataset/archive.zip"
DATA_EXTRACTED = "dataset"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Extract dataset ---
with zipfile.ZipFile(DATA_ZIP, 'r') as zip_ref:
    zip_ref.extractall(DATA_EXTRACTED)

# --- Load dataset ---
df = pd.read_csv(f"{DATA_EXTRACTED}/tidy_format_co2_emission_dataset.csv")
df = df[['Country', 'Year', 'CO2EmissionRate (mt)']].dropna()
df.rename(columns={'CO2EmissionRate (mt)': 'CO2_Emissions'}, inplace=True)

# --- Clean numeric column ---
df['CO2_Emissions'] = (
    df['CO2_Emissions']
    .astype(str)
    .str.replace(',', '')
    .str.replace(' ', '')
    .str.replace('−', '-')
    .str.extract('(\d+\.?\d*)')
)
df['CO2_Emissions'] = pd.to_numeric(df['CO2_Emissions'], errors='coerce')
df = df.dropna(subset=['CO2_Emissions'])

# --- Focus countries ---
countries = ['India', 'United States', 'China', 'Germany']
df = df[df['Country'].isin(countries)]

# --- Train models ---
for country in countries:
    country_data = df[df['Country'] == country].sort_values('Year')
    X = country_data[['Year']]
    y = country_data['CO2_Emissions']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODEL_DIR}/{country}_model.pkl")
    print(f"✅ Model saved for {country}")

print("🎯 All models trained and saved successfully!")
