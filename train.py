#!/usr/bin/env python
# coding: utf-8

# # Problem statement is to predict price column based on data with 24 Columns with over 200 data entries using Linear Regression.

# In[52]:


#import required libraries

# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style
import  missingno as ms

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from scipy import stats 

# Warnings
import warnings
warnings.filterwarnings("ignore")


# In[53]:


#Read data("Data.csv") into dataframe
df = pd.read_csv('Data.csv')
#read df in X
X=df
#Copy Target(column to be predicted) in Y
Y=X.price
#drop target from X, now X is input data
X=X.drop('price', axis=1)
X.head(20)


# In[54]:


X.columns


# In[55]:


#Check for null values in X and Y
print(X.shape)
print(X.isnull().sum())
print(Y.isnull().sum())

#what did you observe?
#ans:- There are no null values in the dataset


# In[56]:


#Check if scaling and encoding are required in X
print(X.info())
X.describe()

#is it required or not?
#ans:- Many of the attributes have non-numeric data, we need to encode them into integers.
# As the range of different attributes is different we need to bring them into common scale


# In[57]:


#Plot relationships between the target variable and any 7 features using pair plot,scatter plot,matrix heatmap
fig, axs = plt.subplots(7, figsize=(10,30))
i=0
compare = 'price'
for feature in df.drop('price',axis=1).columns:
    if i == 7: break
    axs[i].scatter(df[feature], df[compare])
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('price')
    i+=1
    
corr =df.corr()
corr.style.background_gradient(cmap='coolwarm')

#What did you observe?
#ans:- From heatmap we observe a huge correlation between the attributes (like whellbase and carlength etc) 


# In[58]:


cleanup_nums = {"doornumber":     {"four": 4, "two": 2},
                "cylindernumber": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}
X.replace(cleanup_nums, inplace=True)


Cat=X.select_dtypes(include=['object']).copy(deep='False')


Cat=Cat.iloc[:, :].apply(pd.Series)
Name=Cat.CarName.copy()


Temp=[]
Temp=Name.str.split(pat=" ",expand=True)
Temp=Temp[0]
X.CarName=Temp
Cat.CarName=Temp


cleanup_nums = {"CarName":     { "maxda": "mazda" , "porcshce": "porsche" , "Nissan":"nissan" , "vokswagen":"volkswagen", "toyouta" : "toyota","vw" : "volkswagen"} }
X.replace(cleanup_nums, inplace=True)


# In[59]:


#check if One hot encoding is required? if yes do it.
# Instead of one hot encoding which will increase the number of columns in the dataset, we will be using
# label encoding to produce simimlar outcome

# Here as the model of the CarName is unique for each car, hence we are discarding the CarName column and 
# replacing it with the Company column which only useful information
X['Company']=0
for i in range(len(X.CarName)):
    X['Company'][i] = X['CarName'][i].split(' ')[0]
X.drop('CarName', axis=1, inplace=True)
X.shape

# labeling all the object features
labelencoder = LabelEncoder()
target_label = labelencoder.fit_transform(X.Company) 
X.Company = target_label

target_label = labelencoder.fit_transform(X.carbody) 
X.carbody = target_label

target_label = labelencoder.fit_transform(X.fueltype) 
X.fueltype = target_label

target_label = labelencoder.fit_transform(X.aspiration) 
X.aspiration = target_label

target_label = labelencoder.fit_transform(X.drivewheel) 
X.drivewheel = target_label

target_label = labelencoder.fit_transform(X.enginelocation) 
X.enginelocation = target_label

target_label = labelencoder.fit_transform(X.enginetype) 
X.enginetype = target_label

target_label = labelencoder.fit_transform(X.fuelsystem) 
X.fuelsystem = target_label


# In[60]:


#Scale the Dataset
features = X.columns
x=X.loc[:, features].values
x = StandardScaler().fit_transform(x)
X = pd.DataFrame(x)
X.head(10)


# In[94]:


#Splitting data into test and train - 30% Test and 70% Train
X['price'] = Y
train_df, test_df = train_test_split(X)
X_train = train_df.drop('price', axis=1)
Y_train = train_df['price']
X_test = test_df.drop('price', axis=1)
Y_test = test_df['price']


# In[95]:


#Find correlation coeff using linear regression.
lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
# count=0
acc_log = round (lr.score(X_test, Y_test) * 100, 2)
acc_log


# In[96]:


# Print The coefficients
print(lr.coef_)
#What did you observe looking at the coeffients, Describe your observation in atleast 30 words?
#ans:- Positive coeffients imply that the outcome increases with increase in that attribute, simillarly
# negative coffecients immply outcome decreases with increase in that attribute


# In[ ]:




