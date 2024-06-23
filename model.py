import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('Zomato_df.csv')

#print(data.head())


data.drop('Unnamed: 0',axis=1,inplace=True)

x=data.drop('rating',axis=1)
y=data['rating']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=12)


et=ExtraTreesRegressor(n_estimators=120)
et.fit(x_train,y_train)


y_pred=et.predict(x_test)


import pickle
pickle.dump(et,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(y_pred)