import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Binarizer

df=pd.DataFrame({
    "Sno.":[1,2,3,4,5,6,7,8,9,10],
    "Name":['A','B','C','D','E','F','G','H','I','J'],
    "age":[25, 45, 35, 33, 52, 23, 43, 36, 29, 60],
    "salary":[50000, 80000, 60000, 58000, 120000, 52000, 75000, 65000, 48000, 110000],
    "Purchase_amount": [100, 250, 180, 130, 300, 90, 220, 210, 95, 280]
    }
)

print(df)

#transformation
df['purchase%']=(df['Purchase_amount']/df['salary'])*100
print("after transformation: ")
print(df)


 #standardizing/normalizing
df[['salary']]=StandardScaler().fit_transform(df[['salary']])
print("after standardization: ")
print(df)


#Binarization
df[['Purchase_amount']]=Binarizer(threshold=200).fit_transform(df[['Purchase_amount']])
print("after binarization: ")
print(df)


#sampling
sample_df=df.sample(n=4)
print("sample: ")
print(sample_df)


#Discretization
labels=['young', 'adult','senior','super_senior']
df['age_group']=pd.cut(df['age'], bins=4, labels=labels)
print("after discretization: ")
print(df)
