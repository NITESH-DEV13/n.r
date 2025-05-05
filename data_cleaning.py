import pandas as pd
import numpy as np

data = {
    'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', None, 'J'],
    'Age': [25, 30, None, 28, 40, 150, None, 29, 27, 31],
    'Salary': [50000, 60000, 55000, None, 80000, 40000, 1000000, 62000, 58000, 61000],
    'Gender': ['Female', 'male', 'Male', 'Male', 'Female', 'fem', 'Male', 'Male', 'Female', 'female']
}

df=pd.DataFrame(data)

print(df)


#handling outliers
Q1=df['Salary'].quantile(0.25)
Q3=df['Salary'].quantile(0.75)
IQR=Q3-Q1
df=df[(df['Salary']>=Q1-1.5*IQR) & (df['Salary']<=Q3+1.5*IQR)]
print(df)

#handling missing values
# df.dropna(inplace=True)         #dropping rows with null values
df['Name'].fillna('I', inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
print(df)

#handling inconsistent data
df['Gender'] = df['Gender'].str.lower()
df['Gender'] = df['Gender'].replace({'fem': 'female'})
df['Gender'] = df['Gender'].replace({'female': 'Female', 'male': 'Male'})
print(df)