#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#Display all the rows&columns of the dataframe
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


# # Read Dataset

# In[2]:


#train dataset
tr_d=pd.read_csv(r"C:\Users\HP\Desktop\EXLDataSet\Property_Price_Train.csv")
#test dataset
ts_d=pd.read_csv(r"C:\Users\HP\Desktop\EXLDataSet\Property_Price_Test.csv")


# # DATA OVERVIEW

# In[3]:


print(tr_d.shape,ts_d.shape) #shape of dataset with rows and columns


# In[4]:


#tr_d.info()


# In[5]:


#tr_d.head() #top 5 records of dataset


# In[6]:


#tr_d.tail() #last 5 records of dataset


# In[7]:


#tr_d.describe()


# In[8]:


#ts_d.info()


# In[9]:


#ts_d.describe()


# CONCLUSION : The train dataset contains 1459 rows & 81 columns.
#              Sale Price is the target variable and first is id column & remaining 79 are independent variables.
#              Test dataset contains 1459 rows and 80 columns.

# # EXPLORATORY DATA ANALYSIS

# In[10]:


# Check if all variables in both datasets(train,test)are identical except for the response variable("Sale_Price").
(tr_d.columns.drop("Sale_Price")==ts_d.columns).any()


# # Response variable Analysis

# In[11]:


plt.scatter(tr_d['Id'],tr_d['Sale_Price'])
plt.xlabel('Id')
plt.ylabel('Sale Price')
plt.show()


# Above graph gives us the scatter plot of the sale price. Most of the points are assembled on the 
# bottom. And there seems to be no large outliers in the sale price variable.

# In[12]:


#tr_d.Sale_Price.hist(bins=50)
sns.displot(data=tr_d,x="Sale_Price")


# Above graph shows that the distribution of sale prices are right skewed(positive), which shows the distribution 
# of the sale prices isn’t normal. It is reasonable because few people can afford very expensive 
# houses. Need to take transformation to the sale prices variable before model fitting.

# # Numerical Variables

# In[13]:


Numerical_tr=[feature1 for feature1 in tr_d.columns if tr_d[feature1].dtypes!='O'] #O :Object

print('lenght of numerical variables: ',len(Numerical_tr))

tr_d[Numerical_tr].head()


# In[14]:


Numerical_ts=[feature1 for feature1 in ts_d.columns if ts_d[feature1].dtypes!='O'] #O :Object

print('lenght of numerical variables: ',len(Numerical_ts))

ts_d[Numerical_ts].head()  


# In[15]:


#DataFrame of numerical variables
numerical_df_tr = tr_d[Numerical_tr]
numerical_df_ts = ts_d[Numerical_ts]


# In[16]:


numerical_df_tr.hist(figsize=(15,20),bins=30,color="blue",edgecolor="black");


# We notice that there are some columns that center most of their values around a single value. Also some variables are skewed and some variable have a normal/gaussian distribution.

# In[17]:


# drop columns with low variance (since they don’t meaningfully contribute to the model’s predictive capability)

from sklearn.feature_selection import VarianceThreshold

thresholder = VarianceThreshold(threshold=0.15)   # column where 85% of the values are constant
data_high_variance = thresholder.fit(numerical_df_tr)


# In[18]:


# drop column where 85% of the values are constant

high_variance_list = []
for col in numerical_df_tr.columns:
    if col not in numerical_df_tr.columns[thresholder.get_support()]:
        high_variance_list.append(col)

high_variance_list


# In[19]:


tr_d.drop(high_variance_list, axis=1, inplace=True)
ts_d.drop(high_variance_list, axis=1, inplace=True)


# In[20]:


print(tr_d.shape,ts_d.shape)


# # DateTime Variables

# In[21]:


year_tr=[feature1 for feature1 in Numerical_tr if 'Year' in feature1]
print(year_tr)


# In[22]:


year_ts=[feature2 for feature2 in Numerical_ts if 'Year' in feature2]
print(year_ts)


# # Relation between DateTime var and Target Var

# In[23]:


tr_d.groupby('Year_Sold')['Sale_Price'].median().plot()
plt.xlabel('Year_Sold')
plt.ylabel('Median Sale Price')
plt.title('House Price vs Year Sold')


# Here we can see as Sold year increases the sale price of house decreases.

# In[24]:


## Compairing all year variables with target variable
for feature in year_tr:
    if feature!='Year_Sold':
        data1=tr_d.copy()
        #difference between Year_Sold and other year variables
        data1[feature]=data1['Year_Sold']-data1[feature]
        
        plt.scatter(data1[feature],data1['Sale_Price'])
        #plt.xlabel(feature)
        plt.ylabel('Sale Price')
        plt.show()


# Here from above scatter plots we can see that if the difference between year sold and other year variables is high then the sale price is low. It means that if the house is newly build then the price is high and if the house is old then the price is low.

# # Categorical Variable

# In[25]:


categorical_tr=[feature1 for feature1 in tr_d.columns if tr_d[feature1].dtypes=='O']
#categorical_tr


# In[26]:


categorical_ts=[feature1 for feature1 in ts_d.columns if ts_d[feature1].dtypes=='O']
#categorical_ts


# In[27]:


print(len(categorical_tr),len(categorical_ts))


# In[28]:


for feature in categorical_tr:
    print('feature {} : categories {}'.format(feature,tr_d[feature].unique()))


# In[29]:


for feature in categorical_ts:
    print('feature {} : categories {}'.format(feature,ts_d[feature].unique()))


# # Relation between categorical variable and target variable

# In[30]:


for feature in categorical_tr:
    data=tr_d.copy()
    sns.barplot(x=feature,y='Sale_Price',data=data,ci=False,estimator=np.median)
    plt.show()


# Here we can see , some variables are highly dominated by a single category.

# In[31]:


#Dataframe of categorical variables
categorical_df_tr = tr_d[categorical_tr]
categorical_df_ts = ts_d[categorical_ts]


# In[32]:


# variables are highly dominated by one category (more than 90%).

high_dominated_features_tr = []
for col in categorical_df_tr.columns:
    if (categorical_df_tr[col].value_counts().max()/categorical_df_tr[col].count()) > 0.9:
        high_dominated_features_tr.append(col)


# In[33]:


high_dominated_features_tr


# In[34]:


#drop variabkles that are highly dominated by only single category from train dataset
tr_d.drop(high_dominated_features_tr, axis=1, inplace=True)


# In[35]:


#test dataset
high_dominated_features_ts = []
for col in categorical_df_ts.columns:
    if (categorical_df_ts[col].value_counts().max()/categorical_df_ts[col].count()) > 0.9:
        high_dominated_features_ts.append(col)


# In[36]:


#drop variabkles that are highly dominated by only single category from test dataset
ts_d.drop(high_dominated_features_ts, axis=1, inplace=True)


# In[37]:


ts_d=ts_d.drop(['Condition2','Pavedd_Drive'],axis=1)


# In[38]:


print(tr_d.shape,ts_d.shape)


# # Correlation

# In[39]:


#correlation Heatmap
plt.figure(figsize=(25,25))
ax=sns.heatmap(tr_d.corr(),cmap='coolwarm',annot=True,linewidth=2)
#to fix the bug "first and last row cut in half of heatmap"
bottom,top=ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)


# From above heatmap we can see that ,some variables have high correlation with target variable and some are having low correlation. Also some independent variables are showing inter correlation.

# # correlation between target variable and indepedent variables

# In[40]:


list=tr_d[tr_d.columns].corr()['Sale_Price'][:]
tr_d[tr_d.columns].corr()['Sale_Price'][:]


# In[41]:


# variables that have a low correlation with 'SalePrice' [less than 0.25 or -0.25]
condition1  = numerical_df_tr.corr()['Sale_Price'] < 0.25
condition2 = numerical_df_tr.corr()['Sale_Price'] > -0.25
low_corr_cols = (numerical_df_tr.corr()[condition1 & condition2]['Sale_Price'].index).to_list()
low_corr_cols


# In[42]:


# variables that have a high correlation with 'SalePrice'.
high_corr_cols = [elem for elem in (numerical_df_tr.columns).to_list() if elem not in low_corr_cols]
#high_corr_cols


# In[43]:


# drop variables that have a low correlation with 'SalePrice'.
for i in range(len(low_corr_cols)):
    if i in tr_d.columns:
        tr_d.drop(low_corr_cols, axis=1, inplace=True)
        ts_d.drop(low_corr_cols, axis=1, inplace=True)


# In[44]:


print(tr_d.shape,ts_d.shape)


# # DATA PREPROCESSING

# # Data Cleaning

# # Duplicates

# In[45]:


print('number of duplicate values in numerical_df_train dataframe: ',numerical_df_tr.duplicated().sum())
print('number of duplicate values in numerical_df_test dataframe: ',numerical_df_ts.duplicated().sum())
print('number of duplicate values in categorical_df_train dataframe: ',categorical_df_tr.duplicated().sum())
print('number of duplicate values in categorical_df_test dataframe: ',categorical_df_ts.duplicated().sum())


# In[46]:


tr_d.drop_duplicates(inplace=True)
ts_d.drop_duplicates(inplace=True)


# In[47]:


# confirm changes
print('number of duplicate values in train dataframe: ',tr_d.duplicated().sum())
print('number of duplicate values in test dataframe: ',ts_d.duplicated().sum())


# # Year Variables

# In[48]:


#converting year variable into difference between year sold and year variable
#train dataset
for feature in ['Construction_Year', 'Remodel_Year', 'Garage_Built_Year']:
    tr_d[feature]=tr_d['Year_Sold']-tr_d[feature]


# In[49]:


#Relation between total year of the house and sale price
plt.scatter(tr_d['Construction_Year'],tr_d['Sale_Price'])
plt.xlabel('Diff Between Year_Sold & Construction_Year')
plt.ylabel('Sale Price')
plt.show()


# Here from above scatter plots we can see that if the difference between year sold and construction year is high then the sale price is low. It means that if the house is newly build then the price is high and if the house is old then the price is low.

# In[50]:


#test dataset
for feature in ['Construction_Year', 'Remodel_Year', 'Garage_Built_Year']:
    ts_d[feature]=ts_d['Year_Sold']-ts_d[feature]


# In[51]:


tr_d[['Construction_Year', 'Remodel_Year', 'Garage_Built_Year', 'Year_Sold']].head()


# In[52]:


ts_d[['Construction_Year', 'Remodel_Year', 'Garage_Built_Year', 'Year_Sold']].head()


# In[53]:


# renaming the construction year column to total year(difference between Year_Sold and Construction_Year )
tr_d=tr_d.rename(columns={'Construction_Year':'Total_Year'})
ts_d=ts_d.rename(columns={'Construction_Year':'Total_Year'})


# # Drop unnecessary columns

# In[54]:


#Drop the datetime variables
#these variables are showing inter relation
#similar effect on sale price as construction year
tr_d=tr_d.drop(['Remodel_Year','Garage_Built_Year','Garage_Finish_Year','Year_Sold','Month_Sold'],axis=1)
ts_d=ts_d.drop(['Remodel_Year','Garage_Built_Year','Garage_Finish_Year','Year_Sold','Month_Sold'],axis=1)


# In[55]:


#Drop Columns
#these variables are showing inter relation
tr_d=tr_d.drop(['Exterior2nd','Brick_Veneer_Area','Exterior_Material','Basement_Height','Exposure_Level','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','LowQualFinSF','Half_Bathroom_Above_Grade','Garage_Area','W_Deck_Area','Three_Season_Lobby_Area','Screen_Lobby_Area','Pool_Area','Miscellaneous_Value'],axis=1)
ts_d=ts_d.drop(['Exterior2nd','Brick_Veneer_Area','Exterior_Material','Basement_Height','Exposure_Level','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','LowQualFinSF','Half_Bathroom_Above_Grade','Garage_Area','W_Deck_Area','Three_Season_Lobby_Area','Screen_Lobby_Area','Pool_Area','Miscellaneous_Value'],axis=1)


# # Missing Values

# In[56]:


tr_d.isnull().sum()


# In[57]:


sns.heatmap(tr_d.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[58]:


#list of features which has missing values in train dataset
na_features1=[features1 for features1 in tr_d.columns if tr_d[features1].isnull().sum()>=1]
#percentage of missing values
for feature1 in na_features1:
    print(feature1,np.round(tr_d[feature1].isnull().mean(),4),' % missing values')


# In[59]:


#Relation between missing values and target varibale

for feature1 in na_features1:
    data1=tr_d.copy()  #copy of original dataframe
    
    # 1:for missing values and 0:for present values
    data1[feature1]=np.where(data1[feature1].isnull(),1,0)
    
    # to check if there is any relation between missing values and target variable
    # graph of missing value features wrt to target variable i.e. sale price
    #data1.groupby(feature1)['Sale_Price'].median().plot.bar()
    #plt.title(feature1)
    #plt.show()
    sns.barplot(x=feature1,y='Sale_Price',data=data1,ci=False,estimator=np.median)
    plt.show()


# Here we can see the relation between variables with missing values and dependent/target variable. So we have to replace null/missing values also we have to drop those columns which contains missing values more than 50%.

# In[60]:


ts_d.isnull().sum()


# In[61]:


sns.heatmap(ts_d.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[62]:


#list of features which has missing values in test dataset
na_features2=[features2 for features2 in ts_d.columns if ts_d[features2].isnull().sum()>=1]
#percentage of missing values
for feature2 in na_features2:
    print(feature2,np.round(ts_d[feature2].isnull().mean(),4),' % missing values')


# In[63]:


# Drop features which contains missing values greater than 50%
def drop_missing(df):
    i = 0
    for col in df:
        if (df[col].isnull().sum()/1460) > 0.5:
            df.drop(col, axis=1, inplace=True)
            print('column',col,'is dropped')
            i += 1
    if i == 0:
        print('no column dropped')


# In[64]:


drop_missing(tr_d)


# In[65]:


drop_missing(ts_d)


# # Null/Missing value treatment

# In[66]:


def fill_null(df):
    for col in df:
        if (col in Numerical_tr) & (df[col].isnull().any()):
            df[col].fillna(df[col].median(), inplace = True)
            print('fillna numerical column: ',col)
        if (col in categorical_tr) & (df[col].isnull().any()):
            df[col].fillna(df[col].mode().iloc[0], inplace = True)
            print('fillna categorical column: ',col)


# In[67]:


fill_null(tr_d)


# In[68]:


fill_null(ts_d)


# In[69]:


#tr_d.isnull().sum()


# In[70]:


#ts_d.isnull().sum()


# In[71]:


print(tr_d.shape,ts_d.shape)


# # Detect and remove outlires

# In[72]:


#Train Dataset
Numerical_train=[feature1 for feature1 in tr_d.columns if tr_d[feature1].dtypes!='O'] #O :Object

print('lenght of numerical variables: ',len(Numerical_train))


# In[73]:


#DataFrame of numerical variables
numerical_df_train = tr_d[Numerical_train]


# In[74]:


fig, axes = plt.subplots(7, 3, figsize=(18, 50))
i = 0
j = 0
for col in numerical_df_train.columns:
    if j==3:
        i += 1
        j = 0
        sns.boxplot(x=numerical_df_train[col],data=numerical_df_train, palette="Set2", ax=axes[i,j])
    else:
        sns.boxplot(x=numerical_df_train[col],data=numerical_df_train, palette="Set2", ax=axes[i,j])
    j += 1


# In[75]:


Q1 = np.percentile(tr_d['Sale_Price'], 25, interpolation = 'midpoint')
Q3 = np.percentile(tr_d['Sale_Price'], 75, interpolation = 'midpoint')
IQR = Q3 - Q1
# Upper bound
upper = np.where(tr_d['Sale_Price'] >= (Q3+1.5*IQR))
# lower bound
lower = np.where(tr_d['Sale_Price'] <= (Q1-1.5*IQR))
# drop outlires
tr_d.drop(upper[0], errors='ignore', inplace = True)
tr_d.drop(lower[0], errors='ignore', inplace = True)


# In[76]:


#Test Dataset
Numerical_test=[feature1 for feature1 in ts_d.columns if ts_d[feature1].dtypes!='O'] #O :Object

print('lenght of numerical variables: ',len(Numerical_test))


# In[77]:


#DataFrame of numerical variables
numerical_df_test = ts_d[Numerical_test]


# In[78]:


fig, axes = plt.subplots(7, 3, figsize=(18, 50))
i = 0
j = 0
for col in numerical_df_test.columns:
    if j==3:
        i += 1
        j = 0
        sns.boxplot(x=numerical_df_test[col],data=numerical_df_test, palette="Set2", ax=axes[i,j])
    else:
        sns.boxplot(x=numerical_df_test[col],data=numerical_df_test, palette="Set2", ax=axes[i,j])
    j += 1


# In[79]:


# drop outlires
ts_d.drop(upper[0], errors='ignore', inplace = True)
ts_d.drop(lower[0], errors='ignore', inplace = True)


# In[80]:


print(tr_d.shape,ts_d.shape)


# # Converting categorical values to numerical

# In[81]:


categorical_tr1=[feature1 for feature1 in tr_d.columns if tr_d[feature1].dtypes=='O']
categorical_tr1


# In[82]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[83]:


tr_d.Zoning_Class=le.fit_transform(tr_d.Zoning_Class)
tr_d.Property_Shape=le.fit_transform(tr_d.Property_Shape)
tr_d.Land_Outline=le.fit_transform(tr_d.Land_Outline)
tr_d.Lot_Configuration=le.fit_transform(tr_d.Lot_Configuration)
tr_d.Neighborhood=le.fit_transform(tr_d.Neighborhood)
tr_d.Condition1=le.fit_transform(tr_d.Condition1)
tr_d.House_Type=le.fit_transform(tr_d.House_Type)
tr_d.House_Design=le.fit_transform(tr_d.House_Design)
tr_d.Roof_Design=le.fit_transform(tr_d.Roof_Design)
tr_d.Exterior1st=le.fit_transform(tr_d.Exterior1st)
tr_d.Brick_Veneer_Type=le.fit_transform(tr_d.Brick_Veneer_Type)
tr_d.Exterior_Condition=le.fit_transform(tr_d.Exterior_Condition)
tr_d.Foundation_Type=le.fit_transform(tr_d.Foundation_Type)
tr_d.Heating_Quality=le.fit_transform(tr_d.Heating_Quality)
tr_d.Kitchen_Quality=le.fit_transform(tr_d.Kitchen_Quality)
tr_d.Fireplace_Quality=le.fit_transform(tr_d.Fireplace_Quality)
tr_d.Garage=le.fit_transform(tr_d.Garage)
tr_d.Sale_Type=le.fit_transform(tr_d.Sale_Type)
tr_d.Sale_Condition=le.fit_transform(tr_d.Sale_Condition)


# In[84]:


categorical_ts1=[feature1 for feature1 in ts_d.columns if ts_d[feature1].dtypes=='O']
categorical_ts1


# In[85]:


ts_d.Zoning_Class=le.fit_transform(ts_d.Zoning_Class)
ts_d.Property_Shape=le.fit_transform(ts_d.Property_Shape)
ts_d.Land_Outline=le.fit_transform(ts_d.Land_Outline)
ts_d.Lot_Configuration=le.fit_transform(ts_d.Lot_Configuration)
ts_d.Neighborhood=le.fit_transform(ts_d.Neighborhood)
ts_d.Condition1=le.fit_transform(ts_d.Condition1)
ts_d.House_Type=le.fit_transform(ts_d.House_Type)
ts_d.House_Design=le.fit_transform(ts_d.House_Design)
ts_d.Roof_Design=le.fit_transform(ts_d.Roof_Design)
ts_d.Exterior1st=le.fit_transform(ts_d.Exterior1st)
ts_d.Brick_Veneer_Type=le.fit_transform(ts_d.Brick_Veneer_Type)
ts_d.Exterior_Condition=le.fit_transform(ts_d.Exterior_Condition)
ts_d.Foundation_Type=le.fit_transform(ts_d.Foundation_Type)
ts_d.Heating_Quality=le.fit_transform(ts_d.Heating_Quality)
ts_d.Kitchen_Quality=le.fit_transform(ts_d.Kitchen_Quality)
ts_d.Fireplace_Quality=le.fit_transform(ts_d.Fireplace_Quality)
ts_d.Garage=le.fit_transform(ts_d.Garage)
ts_d.Sale_Type=le.fit_transform(ts_d.Sale_Type)
ts_d.Sale_Condition=le.fit_transform(ts_d.Sale_Condition)


# In[86]:


tr_d.dtypes


# In[87]:


ts_d.dtypes


# In[88]:


print(tr_d.shape,ts_d.shape)


# # Skewness

# In[89]:


from scipy.stats import skew


# In[90]:


#Train dataset
for col in tr_d:
    print(col)
    print(skew(tr_d[col]))
    
    plt.figure()
    sns.distplot(tr_d[col])
    plt.show()


# From above graphs we can see that some variables having positive skewness and some are negatively skewed.

# In[91]:


tr_d.skew()


# In[92]:


#skewness treatment
#negative skewness
tr_d["Zoning_Class"]=np.square(tr_d["Zoning_Class"])
tr_d["Land_Outline"]=np.square(tr_d["Land_Outline"])
tr_d["Lot_Configuration"]=np.square(tr_d["Lot_Configuration"])
tr_d["Exterior_Condition"]=np.square(tr_d["Exterior_Condition"])
tr_d["Kitchen_Quality"]=np.square(tr_d["Kitchen_Quality"])
tr_d["Sale_Type"]=np.square(tr_d["Sale_Type"])
tr_d["Sale_Condition"]=np.square(tr_d["Sale_Condition"])


# In[93]:


#Positive skweness
tr_d["Building_Class"]=np.sqrt(tr_d["Building_Class"])
tr_d["Lot_Extent"]=np.sqrt(tr_d["Lot_Extent"])
tr_d["Lot_Size"]=np.sqrt(tr_d["Lot_Size"])
tr_d["Condition1"]=np.sqrt(tr_d["Condition1"])
tr_d["House_Type"]=np.sqrt(tr_d["House_Type"])
tr_d["Roof_Design"]=np.sqrt(tr_d["Roof_Design"])
tr_d["Total_Basement_Area"]=np.sqrt(tr_d["Total_Basement_Area"])
tr_d["First_Floor_Area"]=np.sqrt(tr_d["First_Floor_Area"])
tr_d["Grade_Living_Area"]=np.sqrt(tr_d["Grade_Living_Area"])


# In[94]:


#Test dataset
for col in ts_d:
    print(col)
    print(skew(ts_d[col]))
    
    plt.figure()
    sns.distplot(ts_d[col])
    plt.show()


# In[95]:


sns.distplot(tr_d.Sale_Price)


# After ouliers and skewness treatment Target Variable shows a normal curve.

# In[96]:


ts_d.skew()


# In[97]:


#skewness treatment
#negative skewness
ts_d["Zoning_Class"]=np.square(ts_d["Zoning_Class"])
ts_d["Land_Outline"]=np.square(ts_d["Land_Outline"])
ts_d["Lot_Configuration"]=np.square(ts_d["Lot_Configuration"])
ts_d["Exterior_Condition"]=np.square(ts_d["Exterior_Condition"])
ts_d["Kitchen_Quality"]=np.square(ts_d["Kitchen_Quality"])
ts_d["Sale_Type"]=np.square(ts_d["Sale_Type"])
ts_d["Sale_Condition"]=np.square(ts_d["Sale_Condition"])


# In[98]:


#Positive skweness
ts_d["Building_Class"]=np.sqrt(ts_d["Building_Class"])
ts_d["Condition1"]=np.sqrt(ts_d["Condition1"])
ts_d["House_Type"]=np.sqrt(ts_d["House_Type"])
ts_d["Roof_Design"]=np.sqrt(ts_d["Roof_Design"])
ts_d["First_Floor_Area"]=np.sqrt(ts_d["First_Floor_Area"])
ts_d["Grade_Living_Area"]=np.sqrt(ts_d["Grade_Living_Area"])
ts_d["Open_Lobby_Area"]=np.sqrt(ts_d["Open_Lobby_Area"])
ts_d["Enclosed_Lobby_Area"]=np.sqrt(ts_d["Enclosed_Lobby_Area"])


# In[99]:


print(tr_d.shape,ts_d.shape)


# # Drop Id column

# In[100]:


#Drop Id Column from train dataset
tr_d.drop(["Id"],axis=1,inplace=True)

# Save Id Column to use it in submission file 
id_list=ts_d["Id"].tolist()
#Drop Id column from test dataset
ts_d.drop(["Id"],axis=1,inplace=True)


# # Feature Scaling

# In[101]:


#Feature Scaling Train dataset
feature_scale=[feature for feature in tr_d.columns if feature not in ['Sale_Price']]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(tr_d[feature_scale])


# In[102]:


tr_d=pd.concat([tr_d[['Sale_Price']].reset_index(drop=True),
               pd.DataFrame(scaler.transform(tr_d[feature_scale]),columns=feature_scale)],axis=1)


# In[103]:


#tr_d


# In[104]:


#Feature Scaling test dataset
feature_scale=[feature for feature in ts_d.columns]
scaler.fit(ts_d[feature_scale])


# In[105]:


ts_d=pd.DataFrame(scaler.transform(ts_d[feature_scale]),columns=feature_scale)


# In[ ]:





# # Model Building

# In[106]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# # Train Test Splitting

# In[107]:


# response/target variable
y = tr_d['Sale_Price']
# explanatory/independent variable
X = tr_d.drop('Sale_Price', axis=1)


# In[108]:


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[109]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape 


# # Validation function

# In[110]:


n_folds=10
def RMSE(model):
    kf=KFold(n_folds,shuffle=True,random_state=42).get_n_splits(X_train.values)
    rmse=np.sqrt(-cross_val_score(model,X_train.values,y_train,scoring="neg_mean_squared_error",cv=kf))
    return rmse


# # 1.Linear Reagression

# In[111]:


from sklearn.linear_model import LinearRegression
lmodel = LinearRegression()


# In[112]:


lmodel.fit(X_train,y_train)


# In[113]:


y_pred1 = lmodel.predict(X_test)


# In[114]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[115]:


# MAE
mae1=mean_absolute_error(y_test,y_pred1)
mae1


# In[116]:


# MSE
mse1=mean_squared_error(y_test,y_pred1)
mse1


# In[117]:


# RMSE
rmse1=np.sqrt(mean_squared_error(y_test,y_pred1))
rmse1


# In[118]:


# R^2
rsq1=r2_score(y_test,y_pred1)
rsq1


# In[119]:


rsq11=lmodel.score(X_train,y_train)
# Adjusted R square
adjr1=1-(((1-rsq11)*(1118-1))/(1118-37-1))
adjr1


# In[120]:


#Mean Absolute Percentage Error
e1=y_test-y_pred1
ae1=np.abs(e1) #absolute error
MAPE1=np.mean(ae1/y_test)*100
MAPE1


# In[121]:


#Accuracy
acc1=100-MAPE1
acc1


# In[122]:


#validation
score1=lmodel.score(X_train,y_train)*100
rmse1=RMSE(lmodel).mean()
print("Linear Regression model accuracy score = {:.2f} and mean rmse = {:.3f}".format(score1,rmse1))


# # 2.Random Forest Regressor

# In[123]:


from sklearn.ensemble import RandomForestRegressor
rfmodel = RandomForestRegressor(n_jobs=-1,random_state=0,bootstrap=True)


# In[124]:


rfmodel.fit(X_train,y_train)


# In[125]:


y_pred2 = rfmodel.predict(X_test)


# In[126]:


# MAE
mae2=mean_absolute_error(y_test,y_pred2)
mae2


# In[127]:


# MSE
mse2=mean_squared_error(y_test,y_pred2)
mse2


# In[128]:


# RMSE
rmse2=np.sqrt(mean_squared_error(y_test,y_pred2))
rmse2


# In[129]:


# R^2
rsq2=r2_score(y_test,y_pred2)
rsq2


# In[130]:


rsq22=lmodel.score(X_train,y_train)
# Adjusted R square
adjr2=1-(((1-rsq22)*(1118-1))/(1118-37-1))
adjr2


# In[131]:


#Mean Absolute Percentage Error
e2=y_test-y_pred2
ae2=np.abs(e2) #absolute error
MAPE2=np.mean(ae2/y_test)*100
MAPE2


# In[132]:


#Accuracy
acc2=100-MAPE2
acc2


# In[133]:


#validation
score2=rfmodel.score(X_train,y_train)*100
rmse2=RMSE(rfmodel).mean()
print("Random Forest Model model accuracy score = {:.2f} and mean rmse = {:.3f}".format(score2,rmse2))


# # 3.Lasso

# In[134]:


from sklearn.linear_model import Lasso
lasso=Lasso()


# In[135]:


lasso.fit(X_train,y_train)


# In[136]:


y_pred3 = lasso.predict(X_test)


# In[137]:


# MAE
mae3=mean_absolute_error(y_test,y_pred3)
mae3


# In[138]:


# MSE
mse3=mean_squared_error(y_test,y_pred3)
mse3


# In[139]:


# RMSE
rmse3=np.sqrt(mean_squared_error(y_test,y_pred3))
rmse3


# In[140]:


# R^2
rsq3=r2_score(y_test,y_pred3)
rsq3


# In[141]:


rsq33=lmodel.score(X_train,y_train)
# Adjusted R square
adjr3=1-(((1-rsq3)*(1118-1))/(1118-37-1))
adjr3


# In[142]:


#Mean Absolute Percentage Error
e3=y_test-y_pred3
ae3=np.abs(e3) #absolute error
MAPE3=np.mean(ae3/y_test)*100
MAPE3


# In[143]:


#Accuracy
acc3=100-MAPE3
acc3


# In[144]:


#validation
score3=lasso.score(X_train,y_train)*100
rmse3=RMSE(lasso).mean()
print("Lasso model accuracy score = {:.2f} and mean rmse = {:.3f}".format(score3,rmse3))


# From above results ,we can conclude that using Random Forest Regressor algorithm can prove to be very useful in bringing down the errors (increaing accuracy).

# # Prediction on test dataset

# In[145]:


#ts_d=ts_d.iloc[:,:]
#ts_d.head()


# In[146]:


#final prediction on RandomForestRegressor Model
final_pred=rfmodel.predict(ts_d)
final_pred


# # Submission File

# In[147]:


submission = pd.DataFrame({"Id": id_list,"Sale_Price": final_pred})


# In[148]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




