# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 20:26:21 2018

@author: BDI
"""

import os 
import pandas as pd
import numpy as np
import sklearn

os.chdir(r"C:\Users\BDI\Documents\3-1\데이터관리와 지적경영\분석하자")
mays_df=pd.read_excel('MAYtotal_standard.xls')
mays_df.columns

#금요일 np.array로 변경
X1=np.array(mays_df[['N_weekday_mean', 'N_fri_temp', 'N_fri_rain', 'N_fri_미세',
       'N_fri_초미세', 'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri']])
y1=np.array(mays_df['N_pop_fri'])
print(X1[:3],y1[:3])

#금요일 data split
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test=train_test_split(X1,y1,test_size=0.3,random_state=0)

print(X1_train.shape)
print(y1_train.shape)
print(X1_test.shape)
print(y1_test.shape)

#금요일 Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X1_train,y1_train)
print('Training R^2:', lr.score(X1_train,y1_train))
print('Test R^2:', lr.score(X1_test,y1_test))

#금요일 Coefficient & intercept
m_names=['N_weekday_mean', 'N_fri_temp', 'N_fri_rain', 'N_fri_미세',
       'N_fri_초미세', 'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri']
m_coef=lr.coef_
c_i=pd.Series(m_coef, index=m_names)
c_i=c_i.sort_values(ascending=False)
print(c_i)

#그래프 생성
a_df=pd.DataFrame(c_i)
a_df.to_csv('fri_coef.csv',index=True)

#금요일 KNN-for (n_neighbors=5가 가장 높은 accuracy)
from sklearn.neighbors import KNeighborsRegressor
for i in list(range(1,100)):
    knn=KNeighborsRegressor(n_neighbors=i,weights='uniform')
    knn.fit(X1_train,y1_train)
    print('KNN-',i)
    print('Training Accuracy:', knn.score(X1_train,y1_train))
    print('Test Accuracy:', knn.score(X1_test,y1_test))
    print('\n')

#금요일 KNN
knn=KNeighborsRegressor(n_neighbors=5,weights='uniform')
knn.fit(X1_train,y1_train)

print('Training R^2:', knn.score(X1_train,y1_train))
print('Test R^2:', knn.score(X1_test,y1_test))

#금요일 DT-for (max_depth=10일때 가장 높은 accuracy) 
from sklearn.tree import DecisionTreeRegressor
for i in list(range(1,100)):
    tree = DecisionTreeRegressor(max_depth=i, random_state=0)
    tree.fit(X1_train, y1_train)
    print('DT-',i)
    print('Training Accuracy:', tree.score(X1_train, y1_train))
    print('Test Accuracy:', tree.score(X1_test, y1_test))
    print('\n')

#금요일 DT
tree = DecisionTreeRegressor(max_depth=10, random_state=0)
tree.fit(X1_train, y1_train)
print("훈련데이터 결과:", tree.score(X1_train, y1_train))
print("검증데이터 결과:", tree.score(X1_test, y1_test))

#금요일 RF-for (n_estimators=61일때 가장 높은 accuracy)
from sklearn.ensemble import RandomForestRegressor
for i in list(range(1,100)):
    rf = RandomForestRegressor(n_estimators=i, random_state=0)
    rf.fit(X1_train, y1_train)
    print('RF-',i)
    print('Training Accuracy:', rf.score(X1_train, y1_train))
    print('Test Accuracy:', rf.score(X1_test, y1_test))
    print('\n')
    
#금요일 RF
rf = RandomForestRegressor(n_estimators=61, random_state=0)
rf.fit(X1_train, y1_train)
print("훈련데이터 결과:", rf.score(X1_train, y1_train))
print("검증데이터 결과:", rf.score(X1_test, y1_test))

#금요일 RF 타겟값 정규화 x
mayss_df=pd.read_excel('MAYtotal_standard_둘째주만.xlsx')
mayss_df.columns

#금요일 np.array로 변경
X1=np.array(mayss_df[['N_weekday_mean', 'N_fri_temp', 'N_fri_rain', 'N_fri_미세',
       'N_fri_초미세', 'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri']])
y1=np.array(mayss_df['pop_fri'])
print(X1[:3],y1[:3])

#금요일 data split
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test=train_test_split(X1,y1,test_size=0.3,random_state=0)

print(X1_train.shape)
print(y1_train.shape)
print(X1_test.shape)
print(y1_test.shape)

#금요일 RF for
from sklearn.ensemble import RandomForestRegressor
for i in list(range(1,100)):
    rf = RandomForestRegressor(n_estimators=i, random_state=0)
    rf.fit(X1_train, y1_train)
    print('RF-',i)
    print('Training Accuracy:', rf.score(X1_train, y1_train))
    print('Test Accuracy:', rf.score(X1_test, y1_test))
    print('\n')
    
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(X1_train, y1_train)
print("훈련데이터 결과:", rf.score(X1_train, y1_train))
print("검증데이터 결과:", rf.score(X1_test, y1_test))

#금요일 RF feature importances
rf.feature_importances_
f_names = np.array(['N_weekday_mean', 'N_fri_temp', 'N_fri_rain', 'N_fri_미세',
       'N_fri_초미세', 'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri'])
print(f_names)
f_importances = rf.feature_importances_
for name, importance in zip(f_names, f_importances):
    print(name, importance)
s_importances = pd.Series(f_importances, index=f_names)
s_importances.sort_values(ascending=False)

f_df = pd.DataFrame(s_importances)
f_df.to_excel("f_df.xls", index=True)

"""
일변량통계 변수선택
"""

#금요일 feature selection-일변량통계
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

select=SelectPercentile(percentile=90,score_func=f_regression)
select.fit(X1_train,y1_train)

#선택된 변수 확인
may1_df=mays_df[['N_weekday_mean', 'N_fri_temp', 'N_fri_rain', 'N_fri_미세',
       'N_fri_초미세', 'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri']]
may1_df.columns[np.where(select.get_support()==True)]

#transform을 이용해 선택된 변수만 선택
X1_train_selected=select.transform(X1_train)
X1_train_selected.shape
X1_test_selected=select.transform(X1_test)
X1_test_selected.shape

#LR 비교
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X1_train,y1_train)
print("전체 변수 사용:", lr.score(X1_test, y1_test))
lr.fit(X1_train_selected, y1_train)
X1_test_selected = select.transform(X1_test)
print("선택 변수 사용:", lr.score(X1_test_selected, y1_test))

#KNN 비교
knn=KNeighborsRegressor(n_neighbors=5,weights='uniform')
knn.fit(X1_train_selected,y1_train)

print('Training R^2:', knn.score(X1_train_selected,y1_train))
print('Test R^2:', knn.score(X1_test_selected,y1_test))

#DT 비교
tree = DecisionTreeRegressor(max_depth=10, random_state=0)
tree.fit(X1_train_selected, y1_train)
print("훈련데이터 결과:", tree.score(X1_train_selected, y1_train))
print("검증데이터 결과:", tree.score(X1_test_selected, y1_test))

#RF 비교

rf = RandomForestRegressor(n_estimators=61, random_state=0)
rf.fit(X1_train_selected, y1_train)
print("훈련데이터 결과:", rf.score(X1_train_selected, y1_train))
print("검증데이터 결과:", rf.score(X1_test_selected, y1_test))

"""
성능평가
"""

#금요일 LN 성능평가
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X1_train,y1_train)
pred=lr.predict(X1_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y1_test,pred))
print('MSE:',mean_squared_error(y1_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y1_test,pred)))
print('R^2:',r2_score(y1_test,pred))

#DummyRegressor

from sklearn.dummy import DummyRegressor
dummy=DummyRegressor(strategy='mean')
dummy.fit(X1_train,y1_train)

d_pred=dummy.predict(X1_test)
print('\n-DummyRegressor')
print('MAE:',mean_absolute_error(y1_test,d_pred))
print('MSE:',mean_squared_error(y1_test,d_pred))
print('RMSE:',np.sqrt(mean_squared_error(y1_test,d_pred)))

#금요일 KNN 성능평가 (선택변수)
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=5,weights='uniform')
knn.fit(X1_train_selected,y1_train)
pred=knn.predict(X1_test_selected)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y1_test,pred))
print('MSE:',mean_squared_error(y1_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y1_test,pred)))
print('R^2:',r2_score(y1_test,pred))

#금요일 DT 성능평가
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(max_depth=10, random_state=0)
tree.fit(X1_train,y1_train)
pred=tree.predict(X1_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y1_test,pred))
print('MSE:',mean_squared_error(y1_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y1_test,pred)))
print('R^2:',r2_score(y1_test,pred))

#금요일 RF 성능평가
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(X1_train, y1_train)
pred=rf.predict(X1_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y1_test,pred))
print('MSE:',mean_squared_error(y1_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y1_test,pred)))
print('R^2:',r2_score(y1_test,pred))

#금 LR k-fold CV
from sklearn.model_selection import cross_val_score
lr=LinearRegression()
lr_cv_scores=cross_val_score(lr,X1,y1,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(lr_cv_scores))

#금 KNN k-fold CV
from sklearn.model_selection import cross_val_score
knn=KNeighborsRegressor(n_neighbors=5,weights='uniform')
knn_cv_scores=cross_val_score(knn,X1,y1,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(knn_cv_scores))

#금 DT k-fold CV
from sklearn.model_selection import cross_val_score
tree=DecisionTreeRegressor(max_depth=10, random_state=0)
tree_cv_scores=cross_val_score(tree,X1,y1,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(tree_cv_scores))

#금 RF k-fold CV
from sklearn.model_selection import cross_val_score
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf_cv_scores=cross_val_score(rf,X1,y1,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(rf_cv_scores))

#금요일 순위 뽑아내기**중요(금토일 가장 잘된 모델 하나씩만 할것)
mayorder_df=pd.read_excel('MAYtotal_standard_둘째주만.xlsx')
mayorder_df.columns
X1=np.array(mayorder_df[['N_weekday_mean', 'N_fri_temp', 'N_fri_rain', 'N_fri_미세',
       'N_fri_초미세', 'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri']])
y1=np.array(mayorder_df['pop_fri'])
print(X1[:3],y1[:3])

from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test=train_test_split(X1,y1,test_size=0.3,random_state=0)

print(X1_train.shape)
print(y1_train.shape)
print(X1_test.shape)
print(y1_test.shape)

rank_df=pd.read_excel('MAYtotal_standard_둘째주만.xlsx')
rank_df.columns
rank_org=rank_df[['station','weekday_mean']]

rank_fri=np.array(rank_df[['N_weekday_mean', 'N_fri_temp', 'N_fri_rain', 'N_fri_미세',
       'N_fri_초미세', 'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri']]) #금RF 변수다넣는게 더높아서 다넣음
rank=pd.DataFrame(rank_org)

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=61, random_state=0)
forest.fit(X1_train,y1_train)
result=pd.DataFrame(forest.predict(rank_fri))

#데이터프레임형성
rank['predicted_fri']=result

#정규화된 예측데이터프레임을 역순서대로 정렬해서 weekdaymean이랑 pop량 붙임
rank=rank.sort_values(['station'],ascending=True)
rank=pd.concat([rank,rank_df.iloc[:,-1:]],axis=1)
rank.head()

#증감률 계산
rank['change_rate_fri']=(rank['predicted_fri']-rank['weekday_mean'])/rank['weekday_mean']
rank=rank.sort_values(['change_rate_fri'], ascending=True)
rank.iloc[0:100,:].to_excel('둘째주_금예측_그래프화필요.xls')


