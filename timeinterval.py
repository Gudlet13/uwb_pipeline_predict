import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load your dataset with specific column names and semicolon as delimiter
column_names = ['datetime', 'identity', 'mac_address', 'movement_vector', 'anchor_distances_A1', 'anchor_distances_A2', 'anchor_az_angles_A1', 'anchor_az_angles_A2', 'anchor_el_angles_A1', 'anchor_el_angles_A2']
data = pd.read_csv("formated_event.csv", delimiter=';', skiprows=1, names=column_names)

# Remove "UTC" from the datetime column
data['datetime'] = data['datetime'].str.replace('UTC', '')

# Convert the datetime column to pandas datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Create a binary target variable: 1 if distance >= 25, 0 otherwise
data['stopped'] = (data['anchor_distances_A1'] >= 25) | (data['anchor_distances_A2'] >= 25)

# Create a new feature representing the 3-hour interval
data['1-hour_interval'] = data['datetime'].dt.hour // 1

# Define features and target variable (excluding datetime)
features = ['identity', 'movement_vector', 'anchor_distances_A1', 'anchor_distances_A2', 'anchor_az_angles_A1', 'anchor_az_angles_A2', 'anchor_el_angles_A1', 'anchor_el_angles_A2', '1-hour_interval']
target = 'stopped'

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Apply one-hot encoding to the 'identity' column
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), ['identity'])],
    remainder='passthrough'
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Choose a model (Random Forest Classifier in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_encoded, y_train)

# Make predictions
predictions = model.predict(X_test_encoded)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Count occurrences of stopped and non-stopped events within each 3-hour interval
count_data = data.groupby(['1-hour_interval', 'stopped']).size().unstack(fill_value=0)

# Plot the results
plt.figure(figsize=(10, 6))
sns.lineplot(data=count_data, x='1-hour_interval', y=1, label='Stopped')
sns.lineplot(data=count_data, x='1-hour_interval', y=0, label='Not Stopped')
plt.title('Occurrences of Stopped and Not Stopped Events by 1-hour Interval')
plt.xlabel('1-hour Interval')
plt.ylabel('Count')
plt.legend()
plt.show()
