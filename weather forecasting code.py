import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Loading the data
data = pd.read_csv('flood_data.csv')

# Defining features and target
features = data.drop('Flood Frequency', axis=1)
target = data['Flood Frequency']

# Data Preprocessing
features = features.fillna(features.mean())
target = target.fillna(target.mean())

# Feature Scaling
sc = StandardScaler()
features = sc.fit_transform(features)

# Splitting the data into train and test sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Building the model
model = LinearRegression()
model.fit(features_train, target_train)

# Predicting the test set results
target_pred = model.predict(features_test)

# Calculating the root mean squared error
rmse = np.sqrt(mean_squared_error(target_test, target_pred)))
print('Root Mean Squared Error:', rmse)

# Testing the model with new data
new_data = pd.DataFrame([[75, 55], [80, 60], [90, 70]], columns=['Rainfall', 'Water Level'])
new_data = sc.transform(new_data)
prediction = model.predict(new_data)
print('Predicted Flood Frequency:', prediction)