#!/usr/bin/env python
# coding: utf-8

# Problem Statement:
# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. 
# The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.
# Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
# 

# In[1]:


## Importing Libraries:

import numpy as np # linear algebra
import pandas as pd # used for data manipulation and analysis
import seaborn as sns # used for data visualisation
import matplotlib.pyplot as plt # used for 2D graphics
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=10,10  # to increase the size of the plot


# In[ ]:





# Hypothesis Generation: Hypothesis generation is an educated “guess” of various factors that are impacting the business problem that needs to be solved using machine learning. In framing a hypothesis, the data scientist must not know the outcome of the hypothesis that has been generated based on any evidence.
# 
# •  City type: It can be expected that stores located in Tier 1 cities should have higher sales because of the higher income levels of people there.
#     
# •  Population Density: Stores located in the densely populated areas can be expected to have higher sales because of more demand.
#     
# •  Store Capacity: Stores which are very big in size should have higher sales.
#     
# •  Brand: Branded products should have higher sales because of higher trust in the customer.
#     
# •  Packaging: Products with good packaging can attract customers and sell more.
#     
# •  Utility: Daily use products should have a higher tendency to sell as compared to the specific use products.
#     
# •  Family Size: More the number of family members, more amount will be spent by a customer to buy products especially utility products.
#     
# •  Past Purchase History: Availablity of this information can help us to determine the frequency of a product being purchased by a user.
#     
# 

# ## A simple evaluation method is a train test dataset where the dataset is divided into a train and a test dataset, then the learning model is trained using the train data and performance is measured using the test data.

# In[2]:


## Importing Data files:

df = pd.read_csv('G:\\Internship\\Technocolabs\\Train.csv') 
print(df)


# In[3]:


## The shape property is used to get a tuple(used to store multiple items in a single variable) representing the dimensionality of the DataFrame:

df.shape   


# In[4]:


## This function returns the first 4/5 rows for the object based on position

df.head()  


# In[5]:


##  Pandas describe() is used to view some basic statistical details like percentile, mean, std etc.

df.describe()


# In[6]:


## To get a concise summary of the dataframe:

df.info()


# In[7]:


## To get the column labels of a data frame:

df.columns


# DATA CLEANING

# In[8]:


## To get the number of missing values in the data set:

df.isnull().sum()


# In[9]:


## To remove null values from all coloumns:

## 
# DataFrame.isna - Indicate missing values.
# DataFrame.notna - Indicate existing (non-missing) values.
# DataFrame.fillna - Replace missing values.
# Series.dropna - Drop missing values.
# Index.dropna - Drop missing indices.


df['Item Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
df['Outlet_Size'] = df['Outlet_Size'].fillna('Medium')


# In[10]:


## To increase the width of the existing notebook:

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# Exploratory Data Analysis

# Univariate Analysis

# countplot() method is used to Show the counts of observations in each categorical bin using bar whereas histograms are used when we have continuous measurements and want to understand the distribution of values.

# In[11]:


## Seaborn distplot lets you show a histogram with a line on it:

sns.distplot(df['Item_Outlet_Sales'])


# We can infer that the above figure is having right skewnwss.

# In[12]:


sns.boxplot(df['Item_Weight'])


# From the above figure we can conclude that most of the items have the weight in range 8 to 16 and the mode is approximately 13.

# In[13]:


sns.countplot(x = 'Outlet_Size', data = df)


# We can say that the number of medium size outlets are the highest and high outlet size ones are minimum.

# In[14]:


sns.countplot(x = 'Outlet_Location_Type', data = df)


# We can view that the Tier 3 outlets are more in the dataset.

# In[15]:


sns.countplot(x = 'Outlet_Type', data = df)


# Most of the outlets are Supermarkets of type 1 from the above graph.
# 
# 
# 
# 

# BIVARIATE ANALYSIS

# In[16]:


## regplot() performs a simple linear regression model fit and plot:

sns.regplot(x = 'Item_Weight', y = 'Item_Outlet_Sales', data = df)


# The figure shows that the Item_Outlet_Sales and Item_Weight has less co - relation. Hence linear regression will not work over here.

# In[17]:


sns.regplot(x = 'Item_Visibility', y = 'Item_Outlet_Sales', data = df)


# The above graph shows that there is co - relation between Item_Visibility and Item_Outlet_Sales and the visibility range between 0.00 to 0.20(approximately) has maximum outlet sales.

# In[18]:


sns.boxplot(x = 'Outlet_Identifier', y = 'Item_Outlet_Sales', data = df)


# The above figure shows that OUT027 outlet has the maximum sales and OUT019 and OUT010 has least sales.

# In[19]:


sns.boxplot(x = 'Outlet_Size', y = 'Item_Outlet_Sales', data = df)


# The above figure shows that the Medium sized outlets has more sales.

# In[20]:


sns.boxplot(x = 'Item_Fat_Content', y = 'Item_Outlet_Sales', data = df)


# Large variations in outlet sales with fat content are not observed from the above plot.

# In[21]:


sns.boxplot(x = 'Item_Type', y = 'Item_Outlet_Sales', data = df)


# Here also large variations cannot be observed. Demands for all types of food are there.

# In[22]:


sns.boxplot(x = 'Item_MRP', y = 'Item_Outlet_Sales', data = df)


# No conclusion can be drawn from this graph. Hence we go with pairwise correlation.

# In[23]:


## corr() is used to find the pairwise correlation(table giving correlations that are computed from all observations that have nonmissing values for any pair of variables) of all columns in the dataframe.
## This is used to quantify the degree to which two variables are related.

df.corr()


# In[24]:


sns.heatmap(df.corr())


# In[25]:


## To visualise the relationship among all attributes:

sns.pairplot(df)


# In[26]:


## Since nothing could be concluded from the boxplot we can go with this once:

sns.regplot(x = 'Item_MRP', y = 'Item_Outlet_Sales', data = df)


# The plot is almost linear and shows that higher the Item_MRP higher is the sale.

# FEATURE ENGINEERING

# CONVERTING CATEGORICAL DATA INTO NUMNERICAL

# In[27]:


## to get the unique values:

df['Item_Fat_Content'].unique()


# In[28]:


def fun(x):
    if x == 'Low Fat' or x == 'LF' or x == 'low fat':
        return(0)
    else:
        return(1)


# In[29]:


# apply allow the users to pass a function and apply it on every single value of the Pandas series.

df['Item_Fat_Content'] = df['Item_Fat_Content'].apply(fun)


# In[30]:


df['Item_Fat_Content'].head()


# In[31]:


df['Item_Type'].unique()


# Since there is nothing to be compared from the array of Item_Type so there is no need of 

# In[32]:


df['Outlet_Size'].unique()


# In[33]:


def fun1(x):
    if x == 'Medium':
        return (0)
    elif x == 'High':
            return(1)
    else:
            return(2)


# In[34]:


df['Outlet_Size'] = df['Outlet_Size'].apply(fun1)


# In[35]:


df['Outlet_Size'].head()


# In[36]:


df['Outlet_Location_Type'].unique()


# In[37]:


def fun2(x):
    if x == 'Tier1':
        return (0)
    elif x == 'Tier2':
        return (1)
    else:
        return(2)
        


# In[38]:


df['Outlet_Location_Type'] = df['Outlet_Location_Type'].apply(fun1)


# In[39]:


df['Outlet_Location_Type'].head()


# In[40]:


df['Outlet_Type'].unique()


# In[41]:


def fun3(x):
    if x == 'Supermarket Type 1':
        return (0)
    elif x == 'Supermarket Type 2':
        return (1)
    else:
        return(3)
    


# In[42]:


df['Outlet_Type'] = df['Outlet_Type'].apply(fun2)


# In[43]:


df['Outlet_Type'].head()


# In[44]:


df['Outlet_Identifier'].unique()


# ONE HOT ENCODING - One Hot Encoding is a process in the data processing that is applied to categorical data, to convert it into a binary vector representation for use in machine learning algorithms

# In[45]:


df1 = pd.get_dummies(df['Outlet_Identifier'])


# In[46]:


print(df1)


# In[47]:


df = pd.concat([df,df1], axis = 1)


# In[48]:


df.columns


# In[49]:


df.head()


# FEATURE SELECTION : As from the co-relation map we see that 'Item_Identifier', 'Item_type' and 'Outlet_Establishment_year' has no relation with 'Item_Outlet_sales' we will drop these columns and will not use in training out model.

# In[50]:


## Assigning data to X and Y variable:

x = df.drop(['Item_Identifier', 'Item_Type', 'Outlet_Establishment_Year', 'Item_Outlet_Sales'], axis = 1)
y = df['Item_Outlet_Sales']


# In[51]:


x.head()


# In[52]:


y.head()


# In[53]:


## Training and Testing Data:

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score,classification_report,confusion_matrix


# In[54]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# LINEAR REGRESSION is the next step up after correlation. It is used when we want to predict the value of a variable based on the value of another variable. 

# In[110]:


df.astype({"OUT010":'float', "OUT049":'float', "OUT013":'float', "OUT017":'float', "OUT018":'float', "OUT019":'float',"OUT027":'float',"OUT035":'float',"OUT045":'float',"OUT046":'float' }) 


# In[112]:


from sklearn.linear_model import LinearRegression
lrm = LinearRegression()


# In[ ]:


lrm.fit(x_train,y_train)


# In[115]:


lrm.fit(x_train,y_train)


# In[109]:


predicted = lm.predict(x_test)


# MEAN SQUARED ERROR : In statistics, the mean squared error (MSE) or mean squared deviation (MSD) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
# 
# MEAN ABSOLUTE ERROR : In statistics, mean absolute error (MAE) is a measure of errors between paired observations expressing the same phenomenon. 
#     
# ROOT MEAN SQUARED ERROR(RMSE): The root-mean-square deviation (RMSD) or root-mean-square error (RMSE) is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed. 
#     
# 

# EVALUATION OF LINEAR REGRESSION MODEL

# In[ ]:


print("MEAN SQUARED ERROR(MSE)", mean_squared_error(y_test, predicted))
print("MEAN ABSOLUTE ERROR(MAE)", mean_absolue_error(y_test, predicted))
print("ROOT MEAN SQUARED ERROR(RMSE)", np.sqrt(mean_squred_error(y_test, predicted)))
print("SCORE", lrm.score(x_test, y_test))


# The main difference between Regression and Classification algorithms that Regression algorithms are used to predict the continuous values such as price, salary, age, etc. and 
# Classification algorithms are used to predict/Classify the discrete values such as Male or Female, True or False, Spam or Not Spam, etc.
# 
# 

# RANDOM FOREST REGRESSOR - is a supervised learning algorithm. The "forest" it builds, is an ensemble of decision trees, usually trained with the “bagging” method. 
# The general idea of the bagging method is that a combination of learning models increases the overall result.
# 

# In[ ]:


from sklearn.ensemble import RamdomForestRegressor
rfg = RandomForestRegressor()
rfg.fit(x_train, y_train)


# In[ ]:


predicted = rfg.predict(x_test)


# EVALUATION OF RANDO FOREST REGRESSION MODEL

# In[ ]:


print("MEAN SQUARED ERROR(MSE)",mean_squared_error(y_test,predicted))
print("MEAN ABSOLUTE ERROR(MAE)",mean_absolute_error(y_test,predicted))
print("ROOT MEAN SQUARED ERROR(RMSE)",np.sqrt(mean_squared_error(y_test,predicted)))
print("SCORE",rfg.score(x_test,y_test))


# Ada Boost Regression - is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction.

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
abr=AdaBoostRegressor(n_estimators=70)
abr.fit(x_train,y_train)


# In[ ]:


predicted=abr.predict(x_test)


# EVALUATION :

# In[ ]:


print("MEAN SQUARED ERROR(MSE)",mean_squared_error(y_test,predicted))
print("MEAN ABSOLUTE ERROR(MAE)",mean_absolute_error(y_test,predicted))
print("ROOT MEAN SQUARED ERROR(RMSE)",np.sqrt(mean_squared_error(y_test,predicted)))
print("SCORE",abr.score(x_test,y_test))


# BAGGING REGRESSOR: Bootstrap aggregating, also called bagging (from bootstrap aggregating), is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. 

# In[ ]:


from sklearn.ensemble import BaggingRegressor


# In[ ]:


br=BaggingRegressor(n_estimators=30)
br.fit(x_train,y_train)


# In[ ]:


predicted=br.predict(x_test)


# EVALUATION:

# In[ ]:


print("MEAN SQUARED ERROR(MSE)",mean_squared_error(y_test,predicted))
print("MEAN ABSOLUTE ERROR(MAE)",mean_absolute_error(y_test,predicted))
print("ROOT MEAN SQUARED ERROR(RMSE)",np.sqrt(mean_squared_error(y_test,predicted)))
print("SCORE",br.score(x_test,y_test))


# CONCLUSION:

# In[ ]:




