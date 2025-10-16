#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[2]:


df = pd.read_csv('advertising.csv')
df.head()


# In[3]:


df.describe()


# In[4]:


# Checking main assumptions of Linear Regression: linearity, normality, multicollinearity and homoscedasticity.
# 1) Assumption (Linearity)
sns.pairplot(df,x_vars=["TV","Radio","Newspaper"],y_vars= "Sales",kind="reg")


# In[5]:


import matplotlib.pyplot as plt

# 2) Assumption (Multicollinearity):
df_features = df[["TV","Radio","Newspaper"]]
a = df.corr()['Sales'].sort_values(ascending=False)
print(a)

sns.heatmap(data=df_features.corr())
plt.show()


# In[6]:


# extract the features (independent variables)
X = df.drop(columns=['Unnamed: 0', 'Sales'])
print(X[0:10])


# In[7]:


# extract the dependent (target) variable
y = df['Sales']
print(y[0:10])


# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.80)

from sklearn.linear_model import LinearRegression
lregr = LinearRegression()
#from sklearn.linear_model import SGDRegressor
#sgdr_scaled = SGDRegressor()

# train model (Fit linear model) and evaluate model β coefficients
# NON Standardized
print("NON Standardized")
model = lregr.fit(X_train, y_train)
# print model intercept
print("β0 =", model.intercept_)
# print model coefficients
print("[β1,β2,β3] =", model.coef_)

# estimate residuals
# predict
y_pred = model.predict(X_test)
# residuals is the differences between real y values and predicted y values
residuals = y_test - y_pred
print('Residuals:',residuals[:10])
#print(np.mean(residuals))
#print(np.std(residuals))


# In[9]:


from scipy import stats

# 3) Assumption (Normality of error terms/residuals):
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))
std_residuals = np.std(residuals)
print("Standard deviation of Residuals {}".format(std_residuals))
_, p = stats.normaltest(residuals)
print("p-value:",p)

sns.displot(data=residuals,kde=True)
plt.title('Normality of error terms/residuals')
plt.show()

"""
mean_scaled_residuals = np.mean(residuals_scaled)
print("Mean of Scaled Residuals {}".format(mean_scaled_residuals))
std_scaled_residuals = np.std(residuals_scaled)
print("Standard deviation of Scaled Residuals {}".format(std_scaled_residuals))
_, p = stats.normaltest(residuals_scaled)
print("p-value:",p)

sns.displot(data=residuals_scaled,kde=True)
plt.title('Normality of error terms/residuals (scaled)')
plt.show()
"""


# In[10]:


# 4) Homoscedasticity
plt.figure(figsize=(10,5))
sns.lineplot(x=y_pred,y=residuals,marker='o',color='blue')
plt.xlabel('y_pred (predicted values)')
plt.ylabel('Residuals')
plt.ylim(-10,10)
plt.xlim(0,26)
sns.lineplot(x=[0,26],y=[0,0],color='red')
plt.title('Residuals vs fitted values plot for homoscedasticity check')
plt.show()
"""
plt.figure(figsize=(10,5))
sns.lineplot(x=y_scaled_pred,y=residuals_scaled,marker='o',color='blue')
plt.xlabel('y_scaled_pred (predicted values)')
plt.ylabel('Scaled Residuals')
plt.ylim(-1,1)
plt.xlim(0,2.5)
sns.lineplot(x=[0,2.5],y=[0,0],color='red')
plt.title('Scaled Residuals vs fitted values plot for homoscedasticity check')
plt.show()
"""


# In[11]:


# distribution plot of the target variable
sns.displot(y_train, kde=True)

# statistical test
# computing the p-value for the null-hypothesis that this distribution is a normal distribution
from scipy import stats
_, p = stats.normaltest(y_train)
print(p)


# In[12]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# train scaler
X_train_scaled = sc.fit_transform(X_train)
print(X_train_scaled[0:10])
# apply scaler on test set
X_test_scaled = sc.transform(X_test)
#print(X_val_scaled[0:10])
print("Mean of first feature:", np.mean(X_train_scaled[:,0]), "\nStdev of first feature:", np.std(X_train_scaled[:,0]))


# In[13]:


# y - transformation (box cox)
from sklearn.preprocessing import PowerTransformer
pt_bc = PowerTransformer(method='box-cox')
# Target variable values are transformed to be unskewed
# Apply box-cox on training dataset
# and then use .ravel()  tolatten 2D array to 1D array
y_train_unskewed = pt_bc.fit_transform(y_train.to_frame()).ravel()
print(y_train_unskewed[0:10])
_, p = stats.normaltest(y_train_unskewed)
print(p)
sns.displot(y_train_unskewed, kde=True)

# to apply box-cox transformation to test data set to avoid data leakage
y_test_unskewed = pt_bc.transform(y_test.to_frame()).ravel()
print(y_test_unskewed[0:10])


# In[14]:


lregr_scaled = LinearRegression()

# train model (Fit linear model) and evaluate model β coefficients
# Standardized
model_scaled = lregr_scaled.fit(X_train_scaled, y_train_unskewed)
# print model intercept
print("β0 =", model_scaled.intercept_)
# print model coefficients
print("[β1,β2,β3] =", model_scaled.coef_)

# estimate residuals
# predict
y_pred_unskewed = model_scaled.predict(X_test_scaled)

residuals_unskewed =  pt_bc.inverse_transform(y_test_unskewed.reshape(-1, 1)) - pt_bc.inverse_transform(y_pred_unskewed.reshape(-1, 1))
print('Residuals:', residuals_unskewed[:10])
#print(np.mean(residuals_unskewed))
#print(np.std(residuals_unskewed))


# In[15]:


# METRICS ON THE ORIGINAL DATASET
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# prediction on test data
# Unstandardized model
y_pred = model.predict(X_test)
print(y_pred[0:10])

# Mean Squared Error (MSE)
MSE = mean_squared_error(y_test, y_pred)
# Root Mean Squared Error (RMSE)
RMSE = np.sqrt(MSE)
r2 = r2_score(y_test, y_pred)
print("MSE:" , MSE, ", RMSE:", RMSE, ", R2:", r2)


# In[16]:


# Standardized model
y_pred_unskewed = model_scaled.predict(X_test_scaled)
print(y_pred_unskewed[0:10])

# Mean Squared Error (MSE)
MSE_scaled = mean_squared_error(y_test_unskewed, y_pred_unskewed)
# Root Mean Squared Error (RMSE)
RMSE_scaled = np.sqrt(MSE_scaled)
r2_scaled = r2_score(y_test_unskewed, y_pred_unskewed)
print("MSE:",MSE_scaled,", RMSE:", RMSE_scaled, ", R2:", r2_scaled)

# REVERSE TRANSFORMATION TO BRING target values to the original scale
y_pred_inverse = pt_bc.inverse_transform(y_pred_unskewed.reshape(-1, 1))
print(y_pred_inverse[0:10])

# Mean Squared Error (MSE)
MSE_inverse = mean_squared_error(y_test, y_pred_inverse)
# Root Mean Squared Error (RMSE)
RMSE_inverse = np.sqrt(MSE_inverse)
r2_inverse = r2_score(y_test, y_pred_inverse)
print("MSE:",MSE_inverse,", RMSE:", RMSE_inverse, ", R2:", r2_inverse)


# In[ ]:




