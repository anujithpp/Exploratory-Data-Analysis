import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# 1. Load the Dataset
print("Downloading JM1 Software Defect Prediction dataset...")
# Using fetch_openml to pull the JM1 dataset (ID 310)
jm1 = fetch_openml(data_id=310, as_frame=True, parser='auto')
df = jm1.frame

# 2. Basic Data Inspection
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
df.info()

# 3. Summary Statistics
print("\n--- Summary Statistics ---")
print(df.describe())

# 4. Checking for Missing Values
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# 5. Quick Visualization: Target Variable Distribution
# The target variable 'defects' is usually boolean (True/False for defective module)
plt.figure(figsize=(6, 4))
sns.countplot(x='class', hue='class', data=df, palette='Set2')
plt.title('Distribution of Defective vs. Non-Defective Modules')
plt.xlabel('Contains Defects')
plt.ylabel('Count')
plt.show()

# 6. Correlation Heatmap
print("\n--- Correlation Heatmap ---")
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# 7. Feature Distributions
print("\n--- Feature Distributions ---")
df[['attr1', 'attr2', 'attr3', 'attr4']].hist(bins=30, figsize=(10, 8), color='skyblue')
plt.suptitle('Distribution of Selected Features')
plt.show()