import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np

# Load your dataset
df = pd.read_csv("your_dataset.csv")  # Replace "your_dataset.csv" with the actual path to your dataset

# Convert the 'day' column to datetime format
df['day'] = pd.to_datetime(df['day'])

# Encode categorical columns (if needed)
label_encoder = LabelEncoder()
df['anchor_mac'] = label_encoder.fit_transform(df['anchor_mac'])

# Extract features and target variable
X = df[['anchor_mac', 'distance', 'day', 'line_of_sight']]
y = df['azimuth']  # You can replace 'azimuth' with the target variable you want to predict

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Optionally, you can perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f'Cross-validated R-squared: {np.mean(cv_scores)}')
