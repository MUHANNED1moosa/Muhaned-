#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 
import warnings


# In[ ]:


import pandas as pd

dataTrain = pd.read_csv('train.csv')

dataTrain.head()


# In[ ]:


type(dataTrain)  #data type


# In[ ]:


dataTrain.shape # check dimension


# In[ ]:


dataTest = pd.read_csv('test.csv')
 
dataTrain.head()


# In[ ]:


dataTest.shape


# In[ ]:


dataTrain.isnull().sum()


# In[ ]:


# Checking for null values in each column and displaying the sum of all null values in each column (Testing Set)
dataTest.isnull().sum()


# In[ ]:


dataTrain = dataTrain.dropna()
dataTest = dataTest.dropna()


# In[ ]:


dataTrain.isnull().sum()


# In[ ]:


dataTrain.shape


# In[ ]:


dataTest.isnull().sum()


# In[ ]:


dataTest.shape


# In[ ]:


dataTrain.dtypes  # checking the data type of every column


# In[ ]:


## EDA (Exploratory Data Analysis)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
corr = dataTrain.corr()  
##This is a pandas DataFrame method that is used to calculate the correlation between variables in the DataFrame.
sns.heatmap(corr,annot=True)
plt.show()


# In[ ]:


dataTrain.describe()  #generate various summary statistics of a DataFrame 


# In[ ]:


dataTrain.describe(include = 'object') #summary statistics for categorical values


# In[ ]:


### Regression/scatter Plot
import seaborn as sns
plt.figure(figsize=(10,6))
sns.regplot(x="Item_MRP", y="Item_Outlet_Sales", data=dataTrain)


# In[ ]:


#Pearson correlation coefficient, and p-value.
from scipy import stats
pearson_coef, p_value = stats.pearsonr(dataTrain['Item_MRP'], dataTrain['Item_Outlet_Sales'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(x="Item_Visibility", y="Item_Outlet_Sales", data=dataTrain)


# In[ ]:


pearson_coef, p_value = stats.pearsonr(dataTrain['Item_Visibility'], dataTrain['Item_Outlet_Sales'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(x="Item_Weight", y="Item_Outlet_Sales", data=dataTrain)


# In[ ]:


pearson_coef, p_value = stats.pearsonr(dataTrain['Item_Weight'], dataTrain['Item_Outlet_Sales'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(x="Outlet_Establishment_Year", y="Item_Outlet_Sales", data=dataTrain)


# In[ ]:


pearson_coef, p_value = stats.pearsonr(dataTrain['Outlet_Establishment_Year'], dataTrain['Item_Outlet_Sales'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[ ]:


# Box plot 
sns.boxplot(x="Outlet_Location_Type", y="Item_Outlet_Sales", data=dataTrain)


# In[ ]:


# Box plot 
sns.boxplot(x="Outlet_Size", y="Item_Outlet_Sales", data=dataTrain)


# In[ ]:


# Box plot 
sns.boxplot(x="Outlet_Type", y="Item_Outlet_Sales", data=dataTrain)


# In[ ]:


plt.figure(figsize=(14, 8))  # Increase figure width to 14 inches
sns.boxplot(x="Item_Type", y="Item_Outlet_Sales", data=dataTrain)
plt.xticks(rotation=45, ha='right')  # Rotate labels and align right
plt.xlabel('Item Type', fontsize=12)  # Adjust font size if necessary
plt.ylabel('Item Outlet Sales', fontsize=12)  # Adjust font size if necessary
plt.tight_layout()  # Adjust the layout
plt.show()


# In[ ]:


# Box plot 
sns.boxplot(x="Outlet_Identifier", y="Item_Outlet_Sales", data=dataTrain)


# In[ ]:


# Box plot 
sns.boxplot(x="Item_Fat_Content", y="Item_Outlet_Sales", data=dataTrain)


# In[ ]:


# Box plot 
sns.boxplot(x="Item_Identifier", y="Item_Outlet_Sales", data=dataTrain)


# In[ ]:


#here we using difrent type to solve this problem but it is sill complex data include wthout important to give predictive for target .
# Identify top N items by sales
top_n = dataTrain.groupby('Item_Identifier')['Item_Outlet_Sales'].sum().sort_values(ascending=False).head(10).index

# Filter data for only top N items
top_items_data = dataTrain[dataTrain['Item_Identifier'].isin(top_n)]

# Create a boxplot for these top items
plt.figure(figsize=(14, 8))
sns.boxplot(x="Item_Identifier", y="Item_Outlet_Sales", data=top_items_data)
plt.title('Box Plot of Item Outlet Sales for Top 10 Items')
plt.show()


# In[ ]:


# Using Exploratory data analysis, few features can be dropped because they had no impact on the  prediction. Those features are removed with the function below.(Training set)
dataTrain.drop(['Item_Visibility', 'Item_Weight', 'Item_Identifier','Outlet_Establishment_Year'], axis = 1, inplace = True)


# In[ ]:


# Same features are removed for testing set since the data will be used to train the model
dataTest.drop(['Item_Visibility', 'Item_Weight', 'Item_Identifier','Outlet_Establishment_Year'], axis = 1, inplace = True)


# In[ ]:


dataTrain.shape


# In[ ]:


dataTest.shape


# In[ ]:


for col in categorical_cols:
    if col in dataTrain.columns and col in dataTest.columns:
        # Ensure the data is of type string
        dataTrain[col] = dataTrain[col].astype(str)
        dataTest[col] = dataTest[col].astype(str)

        # Combine the data from both datasets for fitting
        combined_data = pd.concat([dataTrain[col], dataTest[col]], axis=0).unique()
        labelencoder.fit(combined_data)
        
        # Transform the data in both datasets
        dataTrain[col] = labelencoder.transform(dataTrain[col])
        dataTest[col] = labelencoder.transform(dataTest[col])


# In[ ]:


for col in categorical_cols:
    # Combine the data from both datasets for fitting
    all_categories = pd.concat([dataTrain[col], dataTest[col]], axis=0).unique()
    
    # Fit the LabelEncoder on the combined data
    labelencoder.fit(all_categories)
    
    # Transform the data in both datasets
    dataTrain[col] = labelencoder.transform(dataTrain[col])
    dataTest[col] = labelencoder.transform(dataTest[col])


# In[ ]:


for col in categorical_cols:
    if col in dataTrain.columns and col in dataTest.columns:
        # Ensure the data is of type string
        dataTrain[col] = dataTrain[col].astype(str)
        dataTest[col] = dataTest[col].astype(str)

        # Combine the data from both datasets for fitting
        combined_data = pd.concat([dataTrain[col], dataTest[col]], axis=0).unique()
        labelencoder.fit(combined_data)
        
        # Transform the data in both datasets
        dataTrain[col] = labelencoder.transform(dataTrain[col])
        dataTest[col] = labelencoder.transform(dataTest[col])


# In[ ]:


for col in categorical_cols:
    # Combine the data from both datasets for fitting
    all_categories = pd.concat([dataTrain[col], dataTest[col]], axis=0).unique()
    
    # Fit the LabelEncoder on the combined data
    labelencoder.fit(all_categories)
    
    # Transform the data in both datasets
    dataTrain[col] = labelencoder.transform(dataTrain[col])
    dataTest[col] = labelencoder.transform(dataTest[col])


# In[ ]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Create a label encoder object
labelencoder = LabelEncoder()

# List of categorical columns in the Big Mart Sales dataset
categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

for col in categorical_cols:
    # Combine the data from both datasets for fitting
    all_categories = pd.concat([dataTrain[col], dataTest[col]], axis=0).unique()
    
    # Fit the LabelEncoder on the combined data
    labelencoder.fit(all_categories)
    
    # Transform the data in both datasets
    dataTrain[col] = labelencoder.transform(dataTrain[col])
    dataTest[col] = labelencoder.transform(dataTest[col])


# In[ ]:


# checking
dataTrain.head(10)


# In[ ]:


# checkin also for Test
dataTest.head(10)


# In[ ]:



#We use this to perform and reduce the features between the range -1 and 1. This will help the model to predict clearly and better because it is easy to understand, and if the features are at scales with a large difference, the models will be more closely aligned towards the features with large scales. That's why we did this to make dealing easier, clearer, and with all the features
import scipy.stats as stats
dataTrain = stats.zscore(dataTrain)
dataTest = stats.zscore(dataTest)


# In[ ]:


dataTrain


# In[ ]:


dataTest


# In[ ]:


# Splitting the data into features and target variable for the training set
x_train = dataTrain.drop('Item_Outlet_Sales', axis=1)  # Features
y_train = dataTrain['Item_Outlet_Sales']  # Target variable

# For the test set, we only have features, so we use all columns to define x_test
x_test = dataTest  # Use the entire test set as features


# In[ ]:


from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
linear_model = LinearRegression()

# Fit the model on the training data
linear_model.fit(x_train, y_train)

# Predict the target on the training data (for evaluation)
train_preds_linear = linear_model.predict(x_train)


# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate RMSE for the training data
rmse_train_linear = mean_squared_error(y_train, train_preds_linear, squared=False)
print(f"Linear Regression RMSE on Training Data: {rmse_train_linear}")

# Calculate MAE for the training data
mae_train_linear = mean_absolute_error(y_train, train_preds_linear)
print(f"Linear Regression MAE on Training Data: {mae_train_linear}")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)

# Fit the model on the training data
rf_model.fit(x_train, y_train)

# Predict the target on the training data (for evaluation)
train_preds_rf = rf_model.predict(x_train)


# In[ ]:


# Calculate RMSE for the training data
rmse_train_rf = mean_squared_error(y_train, train_preds_rf, squared=False)
print(f"Random Forest RMSE on Training Data: {rmse_train_rf}")

# Calculate MAE for the training data
mae_train_rf = mean_absolute_error(y_train, train_preds_rf)
print(f"Random Forest MAE on Training Data: {mae_train_rf}")


# In[ ]:


# Making predictions on the test data using Random Forest (if it performed better)
y_pred_test_rf = rf_model.predict(x_test)

# OR using Linear Regression (if it performed better)
# y_pred_test_linear = linear_model.predict(x_test)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


# Multiple Linear Regression
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
train_preds_linear = linear_model.predict(x_train)

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(x_train, y_train)
train_preds_rf = rf_model.predict(x_train)


# In[ ]:


# Calculate RMSE and MAE for Multiple Linear Regression
rmse_linear = mean_squared_error(y_train, train_preds_linear, squared=False)
mae_linear = mean_absolute_error(y_train, train_preds_linear)

# Calculate RMSE and MAE for Random Forest Regressor
rmse_rf = mean_squared_error(y_train, train_preds_rf, squared=False)
mae_rf = mean_absolute_error(y_train, train_preds_rf)

print(f"Linear Regression RMSE: {rmse_linear}")
print(f"Linear Regression MAE: {mae_linear}")
print(f"Random Forest RMSE: {rmse_rf}")
print(f"Random Forest MAE: {mae_rf}")


# In[ ]:


# Compare the performance
if rmse_linear < rmse_rf:
    print("Linear Regression performs better based on RMSE.")
else:
    print("Random Forest performs better based on RMSE.")

if mae_linear < mae_rf:
    print("Linear Regression performs better based on MAE.")
else:
    print("Random Forest performs better based on MAE.")


# In[3]:


pip install nbconvert[webpdf]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




