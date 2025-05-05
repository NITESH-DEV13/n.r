import pandas as pd
import numpy as np
from mlxtend.frequent_patterns 
import apriori, association_rules
from mlxtend.preprocessing 
import TransactionEncoder

dataset= [
    ['bread','butter','jam','milk'],
    ['bread','butter','milk'],
    ['bread','juice','curd'],
    ['bread','milk','juice'],
    ['butter','milk','juice']
]

print(dataset)

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df1 = pd.DataFrame(te_ary, columns=te.columns_)
print(df1)

frequent_itemsets1=apriori(df1, min_support=0.5, use_colnames=True)
rules1=association_rules(frequent_itemsets1, metric='confidence', min_threshold=0.75)
print("Rules for Support=50% and Confidence=75%")
print(frequent_itemsets1)
print(rules1[['antecedents', 'consequents', 'support', 'confidence', 'lift']])