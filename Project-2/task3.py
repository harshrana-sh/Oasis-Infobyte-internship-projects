# ============================================
# SALES PREDICTION USING MACHINE LEARNING
# ============================================

import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

path = kagglehub.dataset_download("bumba5341/advertisingcsv")
print("Dataset Path:", path)

file_path = os.path.join(path, os.listdir(path)[0])
df = pd.read_csv(file_path)

print(df.head())

print(df.info())

plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

X = df[['TV', 'Radio', 'Newspaper']]  
y = df['Sales']                       

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()