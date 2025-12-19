import streamlit as st
import pandas as pd
import numpy as np

from model import train_model

st.set_page_config(
    page_title="House Price Predictor",
    layout="wide"
)

st.title("🏠 House Price Predictor")
st.write(
    "Predict **median house price** for an area using a trained **Linear Regression** model."
)

@st.cache_resource
def load_model():
    return train_model()

model, scaler, metrics, feature_names = load_model()


st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("MAE", round(metrics["MAE"], 3))
c2.metric("MSE", round(metrics["MSE"], 3))
c3.metric("R² Score", round(metrics["R2"], 3))

st.sidebar.header("🔢 Enter Area Details")

st.sidebar.markdown(
)

inputs = {}

inputs["MedInc"] = st.sidebar.number_input(
    "Median Income (MedInc)",
    help="Median household income in this area (in $10,000s). Example: 5 → $50,000",
    min_value=0.5,
    max_value=15.0,
    value=5.0
)

inputs["HouseAge"] = st.sidebar.number_input(
    "House Age (years)",
    help="Average age of houses in the area",
    min_value=1.0,
    max_value=60.0,
    value=20.0
)

inputs["AveRooms"] = st.sidebar.number_input(
    "Average Rooms",
    help="Average number of rooms per house",
    min_value=1.0,
    max_value=15.0,
    value=5.0
)

inputs["AveBedrms"] = st.sidebar.number_input(
    "Average Bedrooms",
    help="Average number of bedrooms per house",
    min_value=1.0,
    max_value=5.0,
    value=2.0
)

inputs["Population"] = st.sidebar.number_input(
    "Population",
    help="Total number of people living in the area",
    min_value=100.0,
    max_value=50000.0,
    value=1500.0
)

inputs["AveOccup"] = st.sidebar.number_input(
    "Average Occupancy",
    help="Average number of people per household",
    min_value=1.0,
    max_value=10.0,
    value=3.0
)

inputs["Latitude"] = st.sidebar.number_input(
    "Latitude",
    help="North–South geographic coordinate (California range)",
    min_value=32.0,
    max_value=42.0,
    value=34.0
)

inputs["Longitude"] = st.sidebar.number_input(
    "Longitude",
    help="East–West geographic coordinate (California range)",
    min_value=-124.0,
    max_value=-114.0,
    value=-118.0
)

inputs["Rooms_per_Household"] = inputs["AveRooms"] / inputs["AveOccup"]
inputs["Bedrooms_per_Room"] = inputs["AveBedrms"] / inputs["AveRooms"]


if st.sidebar.button("💰 Predict House Price"):
    input_df = pd.DataFrame([inputs])[feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.subheader("🏡 Estimated House Price")
    st.success(f"${prediction[0] * 100000:,.2f}")

with st.expander("📘 What do these inputs mean?"):
    st.markdown(
        """
        - **MedInc**: Median income of households in the area  
        - **HouseAge**: Average age of houses  
        - **AveRooms**: Average rooms per house  
        - **AveBedrms**: Average bedrooms per house  
        - **Population**: People living in the area  
        - **AveOccup**: People per household  
        - **Latitude / Longitude**: Location coordinates  
        - **Rooms per Household**: How spacious houses are  
        - **Bedrooms per Room**: Bedroom-to-room ratio  
        """
    )
