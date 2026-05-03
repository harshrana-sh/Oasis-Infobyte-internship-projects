                                              # =====================================================
                                              # Unemployment Rate Analysis (COVID Impact)
                                              # =====================================================

import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("gokulrajkmv/unemployment-in-india")
print("Dataset path:", path)

file_path = os.path.join(path, os.listdir(path)[0])
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()   
df = df.dropna()                      

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df['Year'] = df['Date'].dt.year

before_covid = df[df['Year'] < 2020]['Estimated Unemployment Rate (%)'].mean()
after_covid = df[df['Year'] >= 2020]['Estimated Unemployment Rate (%)'].mean()

print("\nAverage Unemployment Rate:")
print("Before COVID:", before_covid)
print("After COVID:", after_covid)

df_year = df.groupby('Year')['Estimated Unemployment Rate (%)'].mean().reset_index()

plt.figure(figsize=(8,5))
plt.bar(df_year['Year'].astype(str), df_year['Estimated Unemployment Rate (%)'])

plt.title("Average Unemployment Rate by Year (COVID Impact)")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")

plt.grid(axis='y')
plt.show()