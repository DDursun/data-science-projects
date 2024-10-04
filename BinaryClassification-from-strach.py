from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
  
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features
y = adult.data.targets 

X = X.drop(['fnlwgt', 'education-num'],axis=1)

y = y[~X.isna().any(axis=1)]
X = X[~X.isna().any(axis=1)]

columns = list(X.columns)
print(columns)
for col in columns:
	col_list = list(X[col])
	if isinstance(col_list[0], int):
		print('Column name: `{}\', type: integer'.format(col))
		print('')
	else:
		print('Column name: `{}\', type: categorical'.format(col))
		print('\tPossible values: {}'.format(list(set(col_list))))
		print('')

print(X.head())



