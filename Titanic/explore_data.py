import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
train_data = pd.read_csv("Datasets/train.csv")
test_data = pd.read_csv("Datasets/test.csv")

# Overview
print("Train shape:", train_data.shape)
print(train_data.info())
print("--------------------------------------------------------")
print(train_data.describe())

print("--------------------------------------------------------")
print("Test shape:", test_data.shape)
print(test_data.info())
print("--------------------------------------------------------")
print(test_data.describe())
print("--------------------------------------------------------")

# Missing values
print("Missing values in train:\n", train_data.isnull().sum())
print("--------------------------------------------------------")
print("Missing values in test:\n", test_data.isnull().sum())
print("--------------------------------------------------------")

# Value counts for key categorical columns
for col in ['Survived', 'Pclass', 'Sex', 'Embarked']:
    print(f"Train {col} value counts:\n{train_data[col].value_counts()}\n")
print("--------------------------------------------------------")

# Unique values for Embarked
print("Train Embarked unique values:", train_data['Embarked'].unique())
print("Test Embarked unique values:", test_data['Embarked'].unique())
print("--------------------------------------------------------")

# Visualizations
sns.countplot(x='Survived', hue='Sex', data=train_data)
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=train_data)
plt.show()


# Correlation heatmap
numeric_cols = train_data.select_dtypes(include=['int64', 'float64'])
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()
