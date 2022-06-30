#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# In[ ]:


data= pd.read_csv(r"C:\Users\Satyansha\OneDrive - IIT Kanpur\SATYANSHA acads\ssummers\ime\Satyansha_200897_assignment1\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(data.head(5))


# In[ ]:


data.info()


# In[ ]:


lst=list(data)
print(lst)


# In[ ]:


del lst[0]
del lst[4]
del lst[-2]
del lst[-2]
print(lst)


# In[ ]:


for a in lst:
    print(set(data[a]))    


# In[ ]:


data.isnull()


# In[ ]:


data.isnull().sum()


# In[ ]:


remove=['customerID']
data.drop(remove,inplace=True,axis=1)
print(data.head(5))


# In[ ]:


data.duplicated()


# In[ ]:


data.drop_duplicates()


# In[ ]:


data['SeniorCitizen'].describe()


# In[ ]:


data['tenure'].describe()


# In[ ]:


data['MonthlyCharges'].describe()


# In[ ]:


data['TotalCharges'].describe()


# In[ ]:


data.TotalCharges = data.TotalCharges.str.strip().str.replace(' ','00')
data.TotalCharges = data.TotalCharges.str.strip().str.replace('','00')


# In[ ]:


data['TotalCharges']= data['TotalCharges'].astype(float)


# In[ ]:


data['TotalCharges'].describe()


# In[ ]:


#data is cleaned and verified now in terms of missing values, noises, outliers, etc
#now before proceeding towards data reduction, lets plot some graphs for understanding the pattern of churn among customers, before and after data reducyion
data['Partner'] =data['Partner'].map({ 'Yes':True , 'No':False })
print(data.head(5))


# In[ ]:


data['Churn'] =data['Churn'].map({ 'Yes' : True , "No" : False })
print(data.head(5))


# In[ ]:


#plotting hostograms of 
#how many senior citizens are there
#how many partners are there
data['Churn'].value_counts().plot(kind='barh', figsize=(6,3))


# In[ ]:


100*data['Churn'].value_counts()/len(data['Churn'])


# In[ ]:


#26.5% people i.e 1869 people churn
data['Churn'].value_counts()


# In[ ]:


data.hist('tenure')


# In[ ]:


data['SeniorCitizen'].value_counts().plot(kind='barh', figsize=(6,3))


# In[ ]:


data['SeniorCitizen'].value_counts()
#1142 senior citizens are there


# In[ ]:


100*data['SeniorCitizen'].value_counts()/len(data['SeniorCitizen'])


# In[ ]:


for i, predictor in enumerate(data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=data, x=predictor, hue='Churn')


# In[ ]:


data['Churn'] = np.where(data.Churn == 'Yes',1,0)
data.head()


# In[ ]:


sns.lmplot(data=data, x='MonthlyCharges', y='TotalCharges', fit_reg=False)
# monthly charges are increasing then total charges also increase, which shows the positive correlation


# In[ ]:


Mth = sns.kdeplot(data.MonthlyCharges[(data["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(data.MonthlyCharges[(data["Churn"] == 1) ],
                color="Blue", shade = True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')


# In[ ]:


Tot = sns.kdeplot(data.TotalCharges[(data["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(data.TotalCharges[(data["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')


# In[ ]:


tenure = data["tenure"]
monthly_charges = data["MonthlyCharges"]
senior = data["SeniorCitizen"]
total_charges= data["TotalCharges"]
churn=data["Churn"]
data.corr()


# In[ ]:


data['Churn'].replace(to_replace='Yes', value=1, inplace=True)
data['Churn'].replace(to_replace='No',  value=0, inplace=True)
df_dummies = pd.get_dummies(data)
df_dummies.head()


# In[ ]:


corr = df_dummies.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[ ]:


corr = data.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[ ]:


data.describe()


# In[ ]:


contingency= pd.crosstab(df_dummies['SeniorCitizen'], df_dummies['Churn'])


# In[ ]:


print(contingency)
c, p, dof, expected = chi2_contingency(contingency)
print(p)


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")
plt.show()


# In[ ]:


contingency= pd.crosstab(df_dummies['Partner'], df_dummies['Churn'])
print(contingency)
c, p, dof, expected = chi2_contingency(contingency)
print(p)
plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")
plt.show()


# In[ ]:


contingency= pd.crosstab(df_dummies['MonthlyCharges'], df_dummies['Churn'])
print(contingency)
c, p, dof, expected = chi2_contingency(contingency)
print(p)
plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")
plt.show()


# In[ ]:


contingency= pd.crosstab(df_dummies['TotalCharges'], df_dummies['Churn'])
print(contingency)
c, p, dof, expected = chi2_contingency(contingency)
print(p)
plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")
plt.show()


# In[ ]:


contingency= pd.crosstab(df_dummies['tenure'], df_dummies['Churn'])
print(contingency)
c, p, dof, expected = chi2_contingency(contingency)
print(p)
plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")
plt.show()


# In[ ]:


contingency= pd.crosstab(df_dummies['MonthlyCharges'], df_dummies['TotalCharges'])
print(contingency)
c, p, dof, expected = chi2_contingency(contingency)
print(p)
plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")
plt.show()


# In[ ]:


contingency= pd.crosstab(df_dummies['MonthlyCharges'], df_dummies['SeniorCitizen'])
print(contingency)
c, p, dof, expected = chi2_contingency(contingency)
print(p)
plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")
plt.show()


# In[ ]:


contingency= pd.crosstab(df_dummies['MonthlyCharges'], df_dummies['tenure'])
print(contingency)
c, p, dof, expected = chi2_contingency(contingency)
print(p)
plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")
plt.show()


# In[ ]:


contingency= pd.crosstab(df_dummies['TotalCharges'], df_dummies['tenure'])
print(contingency)
c, p, dof, expected = chi2_contingency(contingency)
print(p)
plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")
plt.show()


# In[ ]:




