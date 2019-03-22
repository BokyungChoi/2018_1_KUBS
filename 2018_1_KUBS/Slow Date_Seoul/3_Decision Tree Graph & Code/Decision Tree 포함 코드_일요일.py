# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 13:47:55 2018

@author: 김솔이
"""

import os
import pandas as pd
import numpy as np

os.chdir(r'C:\Users\김솔이\Desktop\데이터관리 Team Project\전처리 데이터+코드')
mays_df=pd.read_excel('MAYtotal_Standard_new.xls')

X3=np.array(mays_df[['N_weekday_mean', 
       'N_sat_temp', 'N_sat_rain','N_sat_미세', 'N_sat_초미세',
        'N_sun_temp', 'N_sun_rain','N_sun_미세', 'N_sun_초미세',
       'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri', 'N_trend_sat','N_pop_fri','N_pop_sat']])   
y3=np.array(mays_df['pop_sun'])


from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test=train_test_split(X3,y3,test_size=0.3,random_state=0)

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

select=SelectPercentile(percentile=50,score_func=f_regression)
select.fit(X3_train,y3_train)

#선택된 변수 확인
may3_df=mays_df[['N_weekday_mean', 
       'N_sat_temp', 'N_sat_rain','N_sat_미세', 'N_sat_초미세',
        'N_sun_temp', 'N_sun_rain','N_sun_미세', 'N_sun_초미세',
       'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri', 'N_trend_sat','N_pop_fri','N_pop_sat']]
may3_df.columns[np.where(select.get_support()==True)]

#transform을 이용해 선택된 변수만 선택
X3_train_selected=select.transform(X3_train)
X3_train_selected.shape
X3_test_selected=select.transform(X3_test)
X3_test_selected.shape
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(max_depth=5
                           ,min_samples_leaf=25
                            ,random_state=0)
tree.fit(X3_train_selected,y3_train)

from sklearn.tree import export_graphviz
dot_data=export_graphviz(tree, out_file=None, impurity=False, filled=True)

from IPython.display import Image
import pydotplus

graph=pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())