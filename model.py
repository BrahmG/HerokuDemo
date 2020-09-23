import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random,string,os,warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('hiring.csv')
df
df['experience']=df['experience'].fillna(value=df['experience'].mode())
df.columns=['experience','test_score','interview_score','salary']
df['test_score']=df['test_score'].fillna(value=df['test_score'].mean())

y=df['salary']
x=df.drop('salary',axis=1)
def f(w):
  word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12}
  return word_dict[w]              

x['experience']=x['experience'].apply(f)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
import pickle
pickle.dump(lr,open('model.pkl','wb'))

print(lr.predict([[1,2.6,3]]))
	