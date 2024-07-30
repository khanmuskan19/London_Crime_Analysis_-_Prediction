#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly_express as px
import matplotlib.pyplot as plt


# In[2]:


# !conda update --all


# In[3]:


data=pd.read_csv(r"C:\Users\Muskan Khan\OneDrive\Documents\JUPYTER\London Crime Analysis & Prediction\London_crime.zip", nrows=1000)
data.head()


# In[4]:


data.describe()


# In[5]:


data.shape


# In[6]:


data_col=data.isnull().all()
print(data_col)


# In[7]:


null_percentile=data.isnull().mean()*100
high_np=null_percentile[null_percentile>50]
print(high_np)


# In[8]:


yearly_basis_crimes=data['year'].value_counts().sort_index()
fig=px.line(yearly_basis_crimes, x=yearly_basis_crimes.index,y=yearly_basis_crimes.values, title='Number of Crimes Per Year')
fig.show()


# In[9]:


# I wanna plot the other columns correlation and visualize them, so first I need to encode them.

from sklearn.preprocessing import LabelEncoder
# label_encoder=LabelEncoder()
label_encoder_major = LabelEncoder()

data['major_category_encoded']=label_encoder_major.fit_transform(data['major_category'])

fig=px.bar(data, x='major_category_encoded', y='borough', title="Crime as per the Region", labels={'major_category_encoded':'Crime Type',
                                                                                          'borough':'Region'})
fig.show()


# In[10]:


fig=px.bar(data, x='major_category_encoded',y='value', title='Crime Type Sucuess Rate',labels={'major_category_encoded':'Crime Type',
                                                                                          'value':'Sucess Rate'})
fig.show()


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x=data[['year', 'month']]
y=data['value']

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.10, random_state=42)



# # Creating Pipeline 

# In[12]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
   ( 'scaler', StandardScaler())
])
    
x_train_piped=pipeline.fit_transform(x_train)


# # Selecting Model

# In[13]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

model=RandomForestRegressor()
# model=LinearRegression()

model.fit(x_train_piped, y_train)


# In[14]:


some_x_tedata=x_test[:5]
some_y_tedata=y_test[:5]

prep_some_tedata=pipeline.transform(some_x_tedata)


# In[15]:


some_te_prediction=model.predict(prep_some_tedata)
print(some_te_prediction)
list(some_y_tedata)


# In[16]:


#comparising as a dataframe:


# In[17]:


some_teComp=pd.DataFrame({
    'Actual':some_y_tedata,
    'Predicted': some_te_prediction
})

print(some_teComp)


# # Accuracy

# In[18]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

my_pred=model.predict(x_train_piped)

msa=mean_absolute_error(y_train,my_pred)
mse=mean_squared_error(y_train, my_pred)
rmse=np.sqrt(mse)

print(msa)
print(mse)
print(rmse)


# # Testing on Test Data

# In[19]:


x_test_piped=pipeline.transform(x_test)

my_test_pred=model.predict(x_test_piped)
compare=pd.DataFrame({'Actual':y_test,
    'Predicted': my_test_pred
    
})
print(compare)


# In[20]:


msa=mean_absolute_error(y_test, my_test_pred)
mse=mean_squared_error(y_test,my_test_pred)
rmse=np.sqrt(mse)

print(msa)
print(mse)
print(rmse)
 
# RandomForest:
# 0.7377531340113014
# 1.191152533043475
# 1.0913993462722409

# Linear Regresson:
# 0.748580992824528
# 1.18070062313629
# 1.0866004892030419


# # Cross Validation

# In[21]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(model, x_train_piped, y_train, scoring= 'neg_mean_squared_error',cv=10)
rmse_score=np.sqrt(-score)
# rmse_score=np.sqrt(score)

print(rmse_score)


# In[22]:


y_min=y_train.min
print(y_min)


# In[23]:


y_train.max()


# In[24]:


from joblib import load, dump
dump(model, 'London Crime Prediction')


# In[25]:


model=load('London Crime Prediction')
features=np.array([[2015, 8]])
model.predict(features)


# In[26]:


# Crossvalidation
# array([0.22512693]) by RandomForest
# array([-40.37338179])


# # Crime Type Classification 

# In[27]:


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


label_encoder=LabelEncoder()
data['major_category']=label_encoder.fit_transform(data['major_category'])
data['minor_category']=label_encoder.fit_transform(data['minor_category'])
data['borough']=label_encoder.fit_transform(data['borough'])

x2=data[['year','month','major_category', 'borough']]
y2=data['minor_category']

x_train2, x_test2, y_train2, y_test2=train_test_split(x2,y2, test_size=0.10, random_state=42)

model=LogisticRegression(max_iter=1000)

model.fit(x_train2, y_train2)


# In[28]:


prediction=model.predict(x_train2[:5])
print(prediction)
# print(y_train2[:5])


# In[29]:


x2_test=x_test2[:5]
y2_test=y_test2[:5]

y2_test_pred=model.predict(x2_test)

compare=pd.DataFrame({
    'Actual':y2_test,
    'Predicted':y2_test_pred
})

print(compare)


# In[30]:


from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

accuracy=accuracy_score(y2_test_pred,y2_test)
precision= precision_score(y2_test,y2_test_pred,average='macro')
recall=recall_score(y2_test,y2_test_pred,average='macro')
f1=f1_score(y2_test,y2_test_pred,average='macro')

print(accuracy)
print(recall)
print(f1)


# # Trend Analysis

# In[31]:


# using matplot

trend_data=data.groupby(['year', 'month'])['value'].sum().reset_index()

plt.figure(figsize=(12,6))
plt.plot(trend_data['value'], marker='o')
plt.title('Crime Trend Over Time')
plt.xlabel('Time (Year-Month)')
plt.ylabel('Number of Crimes')
plt.xticks(range(len(trend_data)), [f'{int(row.year)}-{int(row.month):02d}' for row in trend_data.itertuples()], rotation=90)
plt.tight_layout()
plt.show()


# In[39]:


trend_data=data.groupby(['year', 'month'])['value'].sum().reset_index()
trend_data['year_month']=trend_data.apply(lambda row: f'{int(row.year)}-{int(row.month):2d}',axis=1)

fig=px.line(trend_data, x='year_month', y='value', markers='True',  title='Crime Trend Over Time')

fig.update_layout(
        xaxis_title='Time(Year-Month)',
        yaxis_title='Number of Crimes',
        xaxis=dict(tickangle=-90),
        autosize=False,
        width=800,
        height=400
)

fig.show()


# In[ ]:




