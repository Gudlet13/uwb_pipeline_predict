import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import pandas as pd

# Generate sample data
np.random.seed(42)
num_samples = 1000
historical_latitude = np.random.uniform(10, 40, num_samples)
historical_longitude = np.random.uniform(-120, -100, num_samples)
timestamps = pd.date_range(start='2023-01-01', periods=num_samples, freq='H')

# Create a DataFrame to store the historical data
historical_data = pd.DataFrame({
    'Latitude': historical_latitude,
    'Longitude': historical_longitude,
    'Timestamp': timestamps,
})

# Print and display the historical data
print("Historical Data:")
print(historical_data)

# Create fuzzy variables
latitude_var = ctrl.Antecedent(np.arange(30, 40, 1), 'latitude')
longitude_var = ctrl.Antecedent(np.arange(-120, -100, 1), 'longitude')
time_of_day_var = ctrl.Antecedent(np.arange(0, 24, 1), 'time_of_day')
location_var = ctrl.Consequent(np.arange(1, 11, 1), 'location')

# Create fuzzy sets where we can add all sensors values
latitude_var['low'] = fuzz.trimf(latitude_var.universe, [20, 20, 25])
latitude_var['medium'] = fuzz.trimf(latitude_var.universe, [30, 35, 40])
latitude_var['high'] = fuzz.trimf(latitude_var.universe, [35, 40, 40])

longitude_var['far'] = fuzz.trimf(longitude_var.universe, [-120, -120, -110])
longitude_var['medium'] = fuzz.trimf(longitude_var.universe, [-120, -110, -100])
longitude_var['close'] = fuzz.trimf(longitude_var.universe, [-110, -100, -100])

time_of_day_var['morning'] = fuzz.trimf(time_of_day_var.universe, [0, 0, 12])
time_of_day_var['afternoon'] = fuzz.trimf(time_of_day_var.universe, [6, 12, 18])
time_of_day_var['evening'] = fuzz.trimf(time_of_day_var.universe, [12, 18, 24])

location_var['low'] = fuzz.trimf(location_var.universe, [1, 1, 5])
location_var['high'] = fuzz.trimf(location_var.universe, [5, 10, 10])

# Create rules
rule1 = ctrl.Rule(latitude_var['low'] & longitude_var['far'] & time_of_day_var['morning'], location_var['low'])
rule2 = ctrl.Rule(latitude_var['medium'] & longitude_var['medium'] & time_of_day_var['afternoon'], location_var['high'])
rule3 = ctrl.Rule(latitude_var['high'] & longitude_var['close'] & time_of_day_var['evening'], location_var['high'])

# Create fuzzy system
location_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
location_prediction = ctrl.ControlSystemSimulation(location_ctrl)

# Input data for prediction
input_latitude = 35.5
input_longitude = -105
input_time_of_day = 10  # Morning Afternoon Evening

# Pass inputs to the fuzzy system
location_prediction.input['latitude'] = input_latitude
location_prediction.input['longitude'] = input_longitude
location_prediction.input['time_of_day'] = input_time_of_day

# Compute the result
location_prediction.compute()

# Print the predicted location
predicted_location = location_prediction.output['location']
print("Predicted Location Scale:", predicted_location)

# Print the predicted location and confidence level
confidence_level_low = fuzz.interp_membership(location_var.universe, location_var['low'].mf, predicted_location)
confidence_level_high = fuzz.interp_membership(location_var.universe, location_var['high'].mf, predicted_location)
print("Predicted Location Scale:", predicted_location, "with Low Confidence=:", confidence_level_low, "with High Confidence=:", confidence_level_high)

# Plot latitude, longitude, and predicted location
plt.figure(figsize=(10, 6))
plt.scatter(historical_latitude, historical_longitude, label='Historical Location', color='blue')
plt.scatter(input_latitude, input_longitude, label='Input Location', color='red', marker='X', s=100)
plt.axvline(x=predicted_location, color='green', linestyle='--', label='Predicted Location Scale')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Historical, Input, and Predicted Location Scale')
plt.legend()
plt.show()

# Visualize fuzzy sets (optional)
latitude_var.view()
longitude_var.view()
time_of_day_var.view()
location_var.view()
plt.show()
