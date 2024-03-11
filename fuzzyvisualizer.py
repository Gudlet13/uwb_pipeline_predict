import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import pandas as pd

# Generate sample data
np.random.seed(42)
num_samples = 10
timestamps = pd.date_range(start='2023-01-01', periods=num_samples, freq='D')

# Create a DataFrame to store the historical data with timestamps
historical_data = pd.DataFrame({
    'Timestamp': timestamps,
    'Latitude': np.random.uniform(30, 40, num_samples),
    'Longitude': np.random.uniform(-120, -100, num_samples)
})

# Print and display the historical data
print("Dummy Historical Data:")
print(historical_data)

# Create fuzzy variables
latitude_var = ctrl.Antecedent(np.arange(30, 40, 1), 'latitude')
longitude_var = ctrl.Antecedent(np.arange(-120, -100, 1), 'longitude')
location_var = ctrl.Consequent(np.arange(1, 11, 1), 'location')

# Create fuzzy sets
latitude_var['low'] = fuzz.trimf(latitude_var.universe, [20, 20, 25])
latitude_var['medium'] = fuzz.trimf(latitude_var.universe, [30, 35, 40])
latitude_var['high'] = fuzz.trimf(latitude_var.universe, [35, 40, 40])

longitude_var['far'] = fuzz.trimf(longitude_var.universe, [-120, -120, -110])
longitude_var['medium'] = fuzz.trimf(longitude_var.universe, [-120, -110, -100])
longitude_var['close'] = fuzz.trimf(longitude_var.universe, [-110, -100, -100])

location_var['low'] = fuzz.trimf(location_var.universe, [1, 1, 5])
location_var['high'] = fuzz.trimf(location_var.universe, [5, 10, 10])

# Define temporal patterns
day_of_week = ctrl.Antecedent(np.arange(0, 7, 1), 'day_of_week')
hour_of_day = ctrl.Antecedent(np.arange(0, 24, 1), 'hour_of_day')^

day_of_week['Monday'] = fuzz.trimf(day_of_week.universe, [0, 0, 1])
hour_of_day['Morning'] = fuzz.trimf(hour_of_day.universe, [8, 10, 12])

# Create rules based on temporal patterns
rule1 = ctrl.Rule(latitude_var['low'] & longitude_var['far'] &
                  (fuzz.interp_membership(day_of_week.universe, day_of_week['Monday'].mf, historical_data['Timestamp'].dt.dayofweek) == 1) &
                  (fuzz.interp_membership(hour_of_day.universe, hour_of_day['Morning'].mf, historical_data['Timestamp'].dt.hour) == 1),
                  location_var['low'])

rule2 = ctrl.Rule(latitude_var['medium'] & longitude_var['medium'],
                  location_var['high'])

rule3 = ctrl.Rule(latitude_var['high'] & longitude_var['close'] &
                  (fuzz.interp_membership(day_of_week.universe, day_of_week['Monday'].mf, historical_data['Timestamp'].dt.dayofweek) == 1) &
                  (fuzz.interp_membership(hour_of_day.universe, hour_of_day['Morning'].mf, historical_data['Timestamp'].dt.hour) == 1),
                  location_var['high'])

# Create fuzzy system
location_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
location_prediction = ctrl.ControlSystemSimulation(location_ctrl)

# Input data for prediction
input_latitude = 35.5
input_longitude = -105

# Pass inputs to the fuzzy system
location_prediction.input['latitude'] = input_latitude
location_prediction.input['longitude'] = input_longitude

# Compute the result
location_prediction.compute()

# Print the predicted location
predicted_location = location_prediction.output['location']
print("Predicted Location:", predicted_location)

# Print the predicted location and confidence level
confidence_level_low = fuzz.interp_membership(location_var.universe, location_var['low'].mf, predicted_location)
confidence_level_high = fuzz.interp_membership(location_var.universe, location_var['high'].mf, predicted_location)
print("Predicted Location:", predicted_location, "with Confidence (Low):", confidence_level_low, "with Confidence (High):", confidence_level_high)

# Plot historical latitude, longitude, input location, and predicted location
plt.figure(figsize=(10, 6))

# Plot historical latitude and longitude
plt.scatter(historical_data['Latitude'], historical_data['Longitude'], label='Historical Location', color='blue')
# Plot input location
plt.scatter(input_latitude, input_longitude, label='Input Location', color='red', marker='X', s=100)
# Plot predicted location
plt.axvline(x=predicted_location, color='green', linestyle='--', label='Predicted Location')

# Set labels and legend
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Historical, Input, and Predicted Location')
plt.legend()

# Show the plot
plt.show()

# Visualize fuzzy sets (optional)
latitude_var.view()
longitude_var.view()
location_var.view()
day_of_week.view()
hour_of_day.view()
plt.show()
