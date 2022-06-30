#!/usr/bin/env python
# coding: utf-8

# In[109]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# In[110]:


data= pd.read_csv(r"C:\Users\Satyansha\OneDrive - IIT Kanpur\SATYANSHA acads\ssummers\ime\Satyansha_200897_assignment1\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(data.head(5))


# In[111]:


data.info()


# In[112]:


lst=list(data)
print(lst)


# In[113]:


del lst[0]
del lst[4]
del lst[-2]
del lst[-2]
print(lst)


# In[114]:


for a in lst:
    print(set(data[a]))    


# In[115]:


data.isnull()


# In[116]:


data.isnull().sum()


# In[117]:


remove=['customerID']
data.drop(remove,inplace=True,axis=1)
print(data.head(5))


# In[118]:


data.duplicated()


# In[119]:


data.drop_duplicates()


# In[120]:


data['SeniorCitizen'].describe()


# In[121]:


data['tenure'].describe()


# In[122]:


data['MonthlyCharges'].describe()


# In[123]:


data['TotalCharges'].describe()


# In[124]:


data.TotalCharges = data.TotalCharges.str.strip().str.replace(' ','00')
data.TotalCharges = data.TotalCharges.str.strip().str.replace('','00')


# In[125]:


data['TotalCharges']= data['TotalCharges'].astype(float)


# In[126]:


data['TotalCharges'].describe()


# In[127]:


#data is cleaned and verified now in terms of missing values, noises, outliers, etc
#now before proceeding towards data reduction, lets plot some graphs for understanding the pattern of churn among customers, before and after data reducyion
data['Partner'] =data['Partner'].map({ 'Yes':True , 'No':False })
print(data.head(5))


# In[128]:


data['Churn'] =data['Churn'].map({ 'Yes' : True , "No" : False })
print(data.head(5))


# In[129]:


#plotting hostograms of 
#how many senior citizens are there
#how many partners are there
data['Churn'].value_counts().plot(kind='barh', figsize=(6,3))


# In[130]:


100*data['Churn'].value_counts()/len(data['Churn'])


# In[131]:


#26.5% people i.e 1869 people churn
data['Churn'].value_counts()


# In[132]:


data.hist('tenure')


# In[133]:


data['SeniorCitizen'].value_counts().plot(kind='barh', figsize=(6,3))


# In[134]:


data['SeniorCitizen'].value_counts()
#1142 senior citizens are there


# In[135]:


100*data['SeniorCitizen'].value_counts()/len(data['SeniorCitizen'])


# In[136]:


for i, predictor in enumerate(data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=data, x=predictor, hue='Churn')


# In[137]:


data['Churn'] = np.where(data.Churn == 'Yes',1,0)
data.head()


# In[138]:


sns.lmplot(data=data, x='MonthlyCharges', y='TotalCharges', fit_reg=False)
# monthly charges are increasing then total charges also increase, which shows the positive correlation


# In[139]:


Mth = sns.kdeplot(data.MonthlyCharges[(data["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(data.MonthlyCharges[(data["Churn"] == 1) ],
                color="Blue", shade = True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')


# In[140]:


Tot = sns.kdeplot(data.TotalCharges[(data["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(data.TotalCharges[(data["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')


# In[141]:


tenure = data["tenure"]
monthly_charges = data["MonthlyCharges"]
senior = data["SeniorCitizen"]
total_charges= data["TotalCharges"]
churn=data["Churn"]
data.corr()


# In[142]:


data['Churn'].replace(to_replace='Yes', value=1, inplace=True)
data['Churn'].replace(to_replace='No',  value=0, inplace=True)
df_dummies = pd.get_dummies(data)
df_dummies.head()


# In[143]:


corr = df_dummies.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[144]:


corr = data.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[145]:


data.drop(['StreamingMovies','OnlineBackup','TechSupport','PaperlessBilling','Dependents','Partner','PaymentMethod','DeviceProtection'], axis = 1)


# In[146]:


#removing attributes 'StreamingMovies','OnlineBackup','TechSupport','PaperlessBilling','Dependents','Partner','PaymentMethod','DeviceProtection','MultipleLines'
df_dummies = pd.get_dummies(data.drop(['StreamingMovies','OnlineBackup','TechSupport','PaperlessBilling','Dependents','Partner','PaymentMethod','DeviceProtection','MultipleLines'], axis = 1))
df_dummies.head()
corr = df_dummies.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[147]:


x= data


# In[148]:


data.drop(['StreamingMovies','OnlineBackup','TechSupport','PaperlessBilling','Dependents','Partner','PaymentMethod','DeviceProtection','MultipleLines','gender','StreamingTV','Churn'], axis = 1)


# In[149]:


#removing attributes 'StreamingMovies','OnlineBackup','TechSupport','PaperlessBilling','Dependents','Partner','PaymentMethod','DeviceProtection','MultipleLines','gender','StreamingTV','Churn'
#data.drop(['StreamingMovies','OnlineBackup','TechSupport','PaperlessBilling','Dependents','Partner','PaymentMethod','DeviceProtection','MultipleLines','gender','StreamingTV','Churn'], axis = 1)
df_dummies = pd.get_dummies(data.drop(['StreamingMovies','OnlineBackup','TechSupport','PaperlessBilling','Dependents','Partner','PaymentMethod','DeviceProtection','MultipleLines','gender','StreamingTV','Churn'], axis = 1))
df_dummies.head()
data=df_dummies
corr = df_dummies.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[150]:


#data['Partner'] =data['Partner'].map({ 'Yes' : True , "No" : False })
#data['Partner'] = np.where(data.Partner == 'True',1,0)
#print(data.head(5))


# In[151]:


#data['Churn'].replace(to_replace='Yes', value=1, inplace=True)
#data['Churn'].replace(to_replace='No',  value=0, inplace=True)
#print(data.head(5))


# In[152]:


df_dummies.info()


# In[153]:



df_dummies = pd.get_dummies(data)
df_dummies.head()
df_max_scaled = df_dummies.copy()
  
for column in df_max_scaled.columns:
    df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max()
      
# view normalized data
display(df_max_scaled)


# In[154]:


corr = df_dummies.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
#plotting the data correlation, notice that no change is observed, hence data normalization wont cause any harm to the data


# In[155]:


plt.figure(figsize=(12,8)) 
sns.heatmap(data, annot=True, cmap="YlGnBu")


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")
plt.show()


# In[165]:


axes = plt.subplots(1,2, sharey=True, figsize=(6,4))
sns.boxplot(data=data['TotalCharges'], ax=axes[0]);


# In[160]:


plt.scatter(data['TotalCharges'], data['tenure']);


# In[163]:


plt.scatter(data['TotalCharges'], data['MonthlyCharges']);


# In[167]:


plt.scatter(data['TotalCharges'], data['SeniorCitizen']);


# In[168]:


plt.scatter(data['MonthlyCharges'], data['tenure']);


# In[169]:


plt.scatter(data['MonthlyCharges'], data['SeniorCitizen']);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




