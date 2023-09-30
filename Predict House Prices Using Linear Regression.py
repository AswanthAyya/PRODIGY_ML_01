#!/usr/bin/env python
# coding: utf-8

# <center> <h1>Problem Statement</h1> </center>
# 
# <p>Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.
# </p>
# 
# <a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data">Dataset Kaggle Link</a>
# 

# # Importing Necessary Libraries

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import plotly.express as px
import plotly.graph_objects as go


# In[4]:


houses_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# # Data Preprocessing
# ## Data Cleaning and Preparation

# In[5]:


# Display basic information about the dataset
print("Step 2: Data Collection")
print("\nDataset loaded successfully.")
print("\nBasic Information About the Dataset:")
print("Number of Rows:", len(houses_data))
print("Number of Columns:", len(houses_data.columns))
print("\nSample Data (first 5 rows):")
print(houses_data.head())


# In[6]:


houses_data.info()


# In[7]:


houses_data.describe()


# In[8]:


houses_data.isnull().sum()


# In[9]:


# Extract columns with null values
columns_with_null = houses_data.columns[houses_data.isnull().any()]

# Display columns with null values
print("Columns with null values:")
for col in columns_with_null:
    print(col)


# In[10]:


# Check for missing values in each column
missing_values = houses_data.isnull().sum()

# Print columns with missing values and their corresponding counts
columns_with_missing_values = missing_values[missing_values > 0]
print("\nColumns with Missing Values:")
print(columns_with_missing_values)


# In[11]:


# Check for duplicate rows
duplicates_before = houses_data.duplicated().sum()

# Remove duplicate rows
houses_data.drop_duplicates(inplace=True)

# Check for duplicate rows after removal
duplicates_after = houses_data.duplicated().sum()

# Print the results
if duplicates_before > 0:
    print(f"Handling Duplicates\n{duplicates_before} duplicate row(s) were found and removed.")
else:
    print("Handling Duplicates\nNo duplicate rows found in the dataset.")


# In[12]:


# Get the column names and data types
column_info = houses_data.dtypes

# Display column names and data types horizontally
for col_name, data_type in column_info.items():  # Use items() instead of iteritems()
    print(f"{col_name}: {data_type}\t", end='')


# In[13]:


houses_data.columns


# In[14]:


fig = px.scatter_3d(houses_data,x = 'LotArea', y='LotFrontage',z = 'SalePrice')
fig.show()


# In[15]:


#  list of categorical columns to exclude
categorical_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# Select numerical columns
numerical_columns = [col for col in houses_data.columns if col not in categorical_columns]

# Create a DataFrame with only numerical features and the target variable
numerical_data = houses_data[numerical_columns + ['SalePrice']]

# Calculate the correlation matrix
correlation_matrix = numerical_data.corr()

# Step 2: Generate a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix[['SalePrice']], annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation with SalePrice")
plt.show()


# In[16]:


# Split the data into training and testing sets
X = houses_data[['TotalBsmtSF', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath',"FullBath", "HalfBath"]]
y = houses_data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


# Select the columns of interest (features and target)
features = houses_data[['TotalBsmtSF', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']]
target = houses_data[['SalePrice']]

# Create a new DataFrame with only the selected columns
data_subset = pd.concat([features, target], axis=1)  # Use square brackets and specify axis=1

# Calculate the correlation matrix
correlation_matrix = data_subset.corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation Heatmap: Features vs. Target")
plt.show()


# In[18]:


# Check for missing values in the training dataset
missing_train = houses_data.isnull().sum()
print("Missing Values in Training Data:")
print(missing_train)

# Check for missing values in the testing dataset
missing_test = test_data.isnull().sum()
print("\nMissing Values in Testing Data:")
print(missing_test)


# In[19]:


# Missing values in selected features
features_missing_values = X.isnull().sum()
features_missing_values


# In[20]:


#create a linear regression model
model = LinearRegression()


# In[21]:


# Fit the model to the training data
model.fit(X_train, y_train)
print(model)


# In[22]:


# Make predictions on the test data
y_pred = model.predict(X_test)


# In[23]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[24]:


print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[25]:


X.sample(5)


# In[26]:


# Predict the price of a new house
new_house = np.array([[2500, 3, 1,0,2,1]]) 
predicted_price = model.predict(new_house)
print(f"Predicted Price for the New House: ${predicted_price[0]:.2f}")


# In[27]:


# Cross-validation to assess model performance
cv_scores = cross_val_score(model, X, y, cv=5)
print('Cross-Validation Scores:', cv_scores)
print('Mean CV Score:', cv_scores.mean())


# In[28]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

