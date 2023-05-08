import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# LOADING DATASET
train_data = pd.read_csv('data/train.csv')

# EXAMINING DATASET

# first five rows
print(train_data.head())
print("----------------------")
# last five rows
print(train_data.tail())
print("----------------------")
# summary statistics
print(train_data.describe())
print("----------------------")
# data type each variable
print(train_data.info())
print("----------------------")

# Check for missing values
missing_values = train_data.isnull().sum()
print(missing_values)
# create heatmap of missing value
sns.heatmap(train_data.isnull(), cbar=False)
plt.show()
print("----------------------")

'''
# Impute the mean value for missing values in "Age"
mean_age = train_data['Age'].mean()
train_data['Age'].fillna(mean_age, inplace=True)
# Drop the "Cabin" variable
train_data.drop('Cabin', axis=1, inplace=True)
# Drop rows with missing values in "Embarked"
train_data.dropna(subset=['Embarked'], inplace=True)
'''


# Encode categorical variables
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])

# Scale numeric variables
scaler = StandardScaler()
train_data['Age'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))
train_data['Fare'] = scaler.fit_transform(train_data['Fare'].values.reshape(-1, 1))