import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset with specific column names and semicolon as delimiter
column_names = ['datetime', 'identity', 'mac_address', 'movement_vector', 'anchor_distances_A1', 'anchor_distances_A2', 'anchor_az_angles_A1', 'anchor_az_angles_A2', 'anchor_el_angles_A1', 'anchor_el_angles_A2']
data = pd.read_csv("formated_event.csv", delimiter=';', skiprows=1, names=column_names)

# Remove "UTC" from the datetime column
data['datetime'] = data['datetime'].str.replace('UTC', '')

# Convert the datetime column to pandas datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Create a binary target variable: 1 if distance >= 25, 0 otherwise
data['stopped'] = (data['anchor_distances_A1'] >= 25) | (data['anchor_distances_A2'] >= 25)

# Define features and target variable (excluding datetime)
features = ['identity', 'movement_vector', 'anchor_distances_A1', 'anchor_distances_A2', 'anchor_az_angles_A1', 'anchor_az_angles_A2', 'anchor_el_angles_A1', 'anchor_el_angles_A2']
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
# Confusion Matrix Plot
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Stopped', 'Stopped'], yticklabels=['Not Stopped', 'Stopped'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance Plot
feature_importances = model.feature_importances_
features = preprocessor.transformers_[0][1].get_feature_names_out(['identity']) + features[1:]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features, palette='viridis')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Plot')
plt.show()