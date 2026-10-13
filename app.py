
import streamlit as st

st.set_page_config(page_title="AI Hydrogen Power Plant")

st.title("AI-Based Hydrogen Power Plant")

st.write("Welcome to the plant simulation dashboard.")

temperature = st.slider("Temperature (°C)", 40, 80, 60)
voltage = st.slider("Voltage (V)", 20, 50, 40)
current = st.slider("Current (A)", 1, 45, 20)

power = voltage * current

st.metric("Calculated Electrical Power", f"{power} W")

if temperature >= 75:
    st.warning("Simulated high-temperature warning")
else:
    st.success("Simulated operating status: Normal")
