import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define Linz coordinates
min_lat, max_lat = 48.2708, 48.3064  # Latitude range for Linz
min_long, max_long = 14.2619, 14.3225  # Longitude range for Linz


# Function to generate random starting points within the specified area (Linz, Austria)
def generate_random_starting_point():
    # Generate random starting point for each entity within Linz
    start_lat = np.random.uniform(min_lat, max_lat)
    start_long = np.random.uniform(min_long, max_long)

    return start_lat, start_long


# Function to predict the next location based on historical data
def predict_next_location(historical_data, timestamp):
    # Select data close to the given timestamp
    selected_data = historical_data.loc[(historical_data['Timestamp'] - timestamp).abs().idxmin()]

    lat_range = np.linspace(min_lat, max_lat, 1000)
    long_range = np.linspace(min_long, max_long, 1000)

    lat_membership = fuzz.gaussmf(lat_range, selected_data['Latitude'], np.std(historical_data['Latitude']))
    long_membership = fuzz.gaussmf(long_range, selected_data['Longitude'], np.std(historical_data['Longitude']))

    rule_lat = lat_membership
    rule_long = long_membership

    aggregated_lat = np.fmax(rule_lat, lat_membership)
    aggregated_long = np.fmax(rule_long, long_membership)

    predicted_lat = fuzz.defuzz(lat_range, aggregated_lat, 'centroid')
    predicted_long = fuzz.defuzz(long_range, aggregated_long, 'centroid')

    return predicted_lat, predicted_long, selected_data['Latitude'], selected_data['Longitude']


# Number of entities
num_entities = 30

# Time interval between data points (in hours)
time_interval = 1

# Generate random historical data for multiple entities
historical_data = pd.DataFrame()
start_time = datetime.now()

for i in range(num_entities):
    # Generate random starting point within Linz
    start_lat, start_long = generate_random_starting_point()

    # Generate historical data for each entity
    entity_data = pd.DataFrame()
    entity_data['Timestamp'] = [start_time + timedelta(hours=i * time_interval) for i in range(100)]
    entity_data['Latitude'] = np.random.normal(loc=start_lat, scale=0.01, size=100)
    entity_data['Longitude'] = np.random.normal(loc=start_long, scale=0.01, size=100)

    # Concatenate entity data to the overall historical data
    historical_data = pd.concat([historical_data, entity_data], ignore_index=True)

# Save the data to a CSV file
historical_data.to_csv('historical_data_linz.csv', index=False)

# Get user input for the timestamp
user_input = input("Enter the desired timestamp (format: 'YYYY-MM-DD HH:mm:ss'): ")
desired_timestamp = datetime.strptime(user_input, '%Y-%m-%d %H:%M:%S')

# Predict the next location based on historical data for the desired timestamp
predicted_lat, predicted_long, actual_lat, actual_long = predict_next_location(historical_data, desired_timestamp)

# Coordinates of 10 key places in Linz (example coordinates, replace with actual coordinates)
key_places = {
    'Place 1': {'Latitude': 48.2900, 'Longitude': 14.2900},
    'Place 2': {'Latitude': 48.1950, 'Longitude': 14.2000},
    'Place 3': {'Latitude': 48.4000, 'Longitude': 14.3100},
    'Place 4': {'Latitude': 48.5050, 'Longitude': 14.3800},
    'Place 5': {'Latitude': 48.1000, 'Longitude': 14.4300},
    # 'Place 6': {'Latitude': 48.3150, 'Longitude': 14.3400},
    # 'Place 7': {'Latitude': 48.3200, 'Longitude': 14.3500},
    # 'Place 8': {'Latitude': 48.3250, 'Longitude': 14.3600},
    # 'Place 9': {'Latitude': 48.3300, 'Longitude': 14.3700},
    # 'Place 10': {'Latitude': 48.3350, 'Longitude': 14.3800},
}

# Plotting
plt.figure(figsize=(10, 8))  # Adjust figure size
plt.plot(np.linspace(min_lat, max_lat, 1000),
         fuzz.gaussmf(np.linspace(min_lat, max_lat, 1000), actual_lat, np.std(historical_data['Latitude'])),
         label='Actual Latitude')
plt.plot(np.linspace(min_long, max_long, 1000),
         fuzz.gaussmf(np.linspace(min_long, max_long, 1000), actual_long, np.std(historical_data['Longitude'])),
         label='Actual Longitude')
plt.scatter(historical_data['Latitude'], historical_data['Longitude'], color='red',
            label='Historical Data Points')  # Scatter plot with both Latitude and Longitude
plt.scatter([predicted_lat], [predicted_long], color='green', marker='x', label='Predicted Next Location')

# Plot key places
for place, coords in key_places.items():
    plt.scatter(coords['Latitude'], coords['Longitude'], label=place, marker='o', s=50)

plt.legend()
plt.title('Linz Area Overview with Key Places')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid(True)
plt.show()

# Display the predicted and actual next locations
