# ===========================
# Data Cleaning & Exploration
# ===========================

import kagglehub
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset_dir = kagglehub.dataset_download("gokulrajkmv/unemployment-in-india")
files = os.listdir(dataset_dir)
print("Files in dataset folder:", files)

csv_file = os.path.join(dataset_dir, files[0])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv(csv_file)

# Drop missing values if any
if df.isnull().values.any():
    df.dropna(inplace=True)

# Drop duplicates if any
if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)
    
df.columns = df.columns.str.strip()

# Rename columns for simplicity
df.rename(columns={
    "Estimated Unemployment Rate (%)": "Unemployed",
    "Estimated Employed": "Employed",
    "Estimated Labour Participation Rate (%)": "Participation"
}, inplace=True)


print("Dataset Shape:", df.shape)
print("\nData Types & Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nRegion Counts:")
print(df['Region'].value_counts())

# Distribution of Unemployment Rate
plt.figure(figsize=(8,5))
sns.histplot(df["Unemployed"], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")
plt.show()

# Compare Unemployment Rate across Regions
plt.figure(figsize=(12,6))
sns.boxplot(x="Region", y="Unemployed", data=df)
plt.xticks(rotation=90)
plt.title("Unemployment Rate by Region")
plt.show()

# Correlation Heatmap for numeric columns
plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Numeric Columns")
plt.show()

# Optional: Pairplot for relationships between numeric features
sns.pairplot(df[['Unemployed', 'Employed', 'Participation']], diag_kind='kde')
plt.show()
