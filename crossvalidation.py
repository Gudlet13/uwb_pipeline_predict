import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset with specific column names and semicolon as delimiter
column_names = ['datetime', 'identity', 'mac_address', 'movement_vector', 'anchor_distances_A1', 'anchor_distances_A2', 'anchor_az_angles_A1', 'anchor_az_angles_A2', 'anchor_el_angles_A1', 'anchor_el_angles_A2']
data = pd.read_csv("formated_event.csv", delimiter=';', skiprows=1, names=column_names)

# Remove "UTC" from the datetime column
data['datetime'] = data['datetime'].str.replace('UTC', '')

# Convert the datetime column to pandas datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Feature engineering
data['day_of_week'] = data['datetime'].dt.dayofweek
data['hour_of_day'] = data['datetime'].dt.hour
data.drop('datetime', axis=1, inplace=True)  # Drop the original datetime column

# Create a binary target variable: 1 if distance >= 25, 0 otherwise
data['stopped'] = (data['anchor_distances_A1'] >= 25) | (data['anchor_distances_A2'] >= 25)

# Label encode the 'identity' column
label_encoder = LabelEncoder()
data['identity'] = label_encoder.fit_transform(data['identity'])

# Define features and target variable (excluding datetime)
features = ['day_of_week', 'hour_of_day', 'identity', 'movement_vector', 'anchor_distances_A1', 'anchor_distances_A2', 'anchor_az_angles_A1', 'anchor_az_angles_A2', 'anchor_el_angles_A1', 'anchor_el_angles_A2']
target = 'stopped'

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42)

# Choose a model (Random Forest Classifier in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Perform cross-validation
model_cv = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model_cv, data[features], data[target], cv=5, scoring='accuracy')

# Display cross-validation scores
print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())
