import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Download dataset
path = kagglehub.dataset_download("vijayaadithyanvg/car-price-predictionused-cars")

# Load data
df = pd.read_csv(os.path.join(path, "car data.csv"))

# Preprocess
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Car Price Prediction")
plt.show()