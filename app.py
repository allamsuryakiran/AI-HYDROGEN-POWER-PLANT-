
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="AI Hydrogen Power Plant",
    page_icon="⚡",
    layout="wide"
)

st.title("AI-Based Hydrogen Power Plant")
st.write("AI-powered simulation for power prediction and plant monitoring.")
st.info("Educational simulation using synthetic data. Not connected to a real plant.")

# 1. Generate simulated training data
rng = np.random.default_rng(42)
n = 1500

temperature = rng.uniform(40, 80, n)
pressure = rng.uniform(1, 3, n)
flow = rng.uniform(5, 20, n)
voltage = rng.uniform(20, 50, n)

current = np.clip(
    2 * flow + 0.5 * pressure
    - 0.03 * (temperature - 60)
    + rng.normal(0, 1.5, n),
    1, None
)

power = voltage * current

df = pd.DataFrame({
    "Temperature": temperature,
    "Pressure": pressure,
    "Hydrogen_Flow": flow,
    "Voltage": voltage,
    "Current": current,
    "Power": power
})

# 2. Train and evaluate the AI model
features = [
    "Temperature", "Pressure", "Hydrogen_Flow",
    "Voltage", "Current"
]

X = df[features]
y = df["Power"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

st.header("Plant Operating Conditions")

# 3. User inputs
c1, c2 = st.columns(2)

with c1:
    temp = st.slider("Temperature (°C)", 40, 80, 60)
    pressure_in = st.slider("Pressure (bar)", 1.0, 3.0, 2.0)
    flow_in = st.slider("Hydrogen flow (simulation units)", 5.0, 20.0, 12.0)

with c2:
    voltage_in = st.slider("Voltage (V)", 20.0, 50.0, 40.0)
    current_in = st.slider("Current (A)", 1.0, 45.0, 20.0)
    leak_flag = st.checkbox("Simulate hydrogen leak warning")

# 4. Predict power
new_data = pd.DataFrame([{
    "Temperature": temp,
    "Pressure": pressure_in,
    "Hydrogen_Flow": flow_in,
    "Voltage": voltage_in,
    "Current": current_in
}])

predicted_power = float(model.predict(new_data)[0])
calculated_power = voltage_in * current_in

st.header("AI Prediction Results")

m1, m2, m3 = st.columns(3)
m1.metric("AI-Predicted Power", f"{predicted_power:.2f} W")
m2.metric("Calculated Power (V × I)", f"{calculated_power:.2f} W")
m3.metric("Temperature", f"{temp} °C")

# 5. Simulated status warnings
if leak_flag:
    st.error("SIMULATED ALERT: Hydrogen leak flag activated.")
elif temp >= 75:
    st.warning("SIMULATED ALERT: High temperature.")
else:
    st.success("Simulated operating status: Normal")

# 6. Power comparison chart
st.header("Power Output Comparison")

chart_df = pd.DataFrame({
    "Power type": ["AI prediction", "V × I calculation"],
    "Power (W)": [predicted_power, calculated_power]
})

st.bar_chart(chart_df.set_index("Power type"))

# 7. AI model evaluation
st.header("AI Model Evaluation")

e1, e2 = st.columns(2)
e1.metric("Test MAE", f"{mae:.2f} W")
e2.metric("Test R²", f"{r2:.3f}")

st.caption(
    "Model evaluation uses held-out synthetic data. "
    "These metrics do not establish real-world plant accuracy."
)

# 8. Temperature versus predicted power
st.header("Temperature Analysis")

temperatures = np.arange(40, 81, 2)

scenario = pd.DataFrame({
    "Temperature": temperatures,
    "Pressure": [pressure_in] * len(temperatures),
    "Hydrogen_Flow": [flow_in] * len(temperatures),
    "Voltage": [voltage_in] * len(temperatures),
    "Current": [current_in] * len(temperatures)
})

curve = model.predict(scenario)

graph_df = pd.DataFrame({
    "Temperature (°C)": temperatures,
    "Predicted power (W)": curve
}).set_index("Temperature (°C)")

st.line_chart(graph_df)

st.header("Project Summary")
st.write(
    "This prototype trains a Random Forest regression model on "
    "synthetic hydrogen power plant data. It predicts electrical "
    "power, compares the prediction with V × I, and demonstrates "
    "simulated operating alerts."
)

