
# coding: utf-8

# In[39]:


import os 
import pandas as pd
import numpy as np
import sklearn


# In[179]:


os.chdir(r"C:\Users\최보경\Desktop\2018JUNIOR\01-데이터관리와지적경영\teamdata\final")


# In[70]:


# In[180]:


mays_df=pd.read_excel('MAYtotal_Standard_new.xls')


# In[181]:


mays_df.columns


# In[182]:


mays_df.head()


# In[101]:

#일요일예측 ndarray로 변경하기
X3=np.array(mays_df[['N_weekday_mean', 
       'N_sat_temp', 'N_sat_rain','N_sat_미세', 'N_sat_초미세',
        'N_sun_temp', 'N_sun_rain','N_sun_미세', 'N_sun_초미세',
       'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri', 'N_trend_sat','N_trend_sun','N_pop_fri','N_pop_sat']])   
y3=np.array(mays_df['pop_sun'])
print(X3[:3],y3[:3])


# In[168]:


#일요일 Data split
from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test=train_test_split(X3,y3,test_size=0.3,random_state=0)

print(X3_train.shape)
print(y3_train.shape)
print(X3_test.shape)
print(y3_test.shape)


# In[169]:


#일요일 feature selection(이걸로 선택!)-일변량통계
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

select=SelectPercentile(percentile=50,score_func=f_regression)
select.fit(X3_train,y3_train)

#선택된 변수 확인
may3_df=mays_df[['N_weekday_mean', 
       'N_sat_temp', 'N_sat_rain','N_sat_미세', 'N_sat_초미세',
        'N_sun_temp', 'N_sun_rain','N_sun_미세', 'N_sun_초미세',
       'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri', 'N_trend_sat','N_trend_sun','N_trend_sun','N_pop_fri','N_pop_sat']]
may3_df.columns[np.where(select.get_support()==True)]


# In[170]:


#transform을 이용해 선택된 변수만 선택
X3_train_selected=select.transform(X3_train)
X3_train_selected.shape
X3_test_selected=select.transform(X3_test)
X3_test_selected.shape


# In[171]:


#일요일 Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X3_train_selected,y3_train)
print('Training R^2:', lr.score(X3_train_selected,y3_train))
print('Test R^2:', lr.score(X3_test_selected,y3_test))


# In[174]:


#일요일 Coefficient & intercept
m_names=['N_weekday_mean', 'N_foodindustry', 'N_buss_density', 'N_univ_num',
       'N_trend_thu', 'N_trend_fri', 'N_trend_sat', 'N_trend_sun',
       'N_pop_fri']
m_coef=lr.coef_
c_i=pd.Series(m_coef, index=m_names)
c_i=c_i.sort_values(ascending=False)
print(c_i)


# In[175]:


##그래프 생성
c_df=pd.DataFrame(c_i)
c_df.to_csv('sun_coef.csv',index=True)
#그래프 첨부


# In[176]:


#일요일 KNN-for (n_neighbors=7가 가장 높은 accuracy)
from sklearn.neighbors import KNeighborsRegressor
for i in list(range(1,50)):
    knn=KNeighborsRegressor(n_neighbors=i,weights='uniform')
    knn.fit(X3_train_selected,y3_train)
    print('KNN-',i)
    print('Training Accuracy:', knn.score(X3_train_selected,y3_train))
    print('Test Accuracy:', knn.score(X3_test_selected,y3_test))
    print('\n')


# In[362]:


#일요일 KNN
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=7,weights='uniform')
knn.fit(X3_train_selected,y3_train)

print('Training R^2:', knn.score(X3_train_selected,y3_train))
print('Test R^2:', knn.score(X3_test_selected,y3_test))


#일요일 Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(max_depth=5
                           ,min_samples_leaf=25
                            ,random_state=0)
tree.fit(X3_train_selected,y3_train)
print('Training R^2:', tree.score(X3_train_selected,y3_train))
print('Test R^2:', tree.score(X3_test_selected,y3_test))


# In[286]:


#일요일 RandomForest
from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=28,max_depth=14,random_state=0)
forest.fit(X3_train_selected,y3_train)
print('Training R^2:', forest.score(X3_train_selected,y3_train))
print('Test R^2::', forest.score(X3_test_selected,y3_test))


# In[287]:


f_importances=forest.feature_importances_

for name, importance in zip(m_names,f_importances):
    print(name, importance)


# In[288]:


#sort
s_importances=pd.Series(f_importances,index=m_names)
s_importances.sort_values(ascending=False)



# In[225]:

#일 LR 성능평가
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X3_train_selected,y3_train)
pred=lr.predict(X3_test_selected)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y3_test,pred))
print('MSE:',mean_squared_error(y3_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y3_test,pred)))
print('R^2:',r2_score(y3_test,pred))


# In[156]:


#일 KNN 성능평가
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=12,weights='uniform')
knn.fit(X3_train_selected,y3_train)
pred=knn.predict(X3_test_selected)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y3_test,pred))
print('MSE:',mean_squared_error(y3_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y3_test,pred)))
print('R^2:',r2_score(y3_test,pred))

#일 dummy 성능평가
from sklearn.dummy import DummyRegressor
dummy=DummyRegressor(strategy='mean')
dummy.fit(X3_train_selected,y3_train)

d_pred=dummy.predict(X3_test_selected)
print('\n-DummyRegressor')
print('MAE:',mean_absolute_error(y3_test,d_pred))
print('MSE:',mean_squared_error(y3_test,d_pred))
print('RMSE:',np.sqrt(mean_squared_error(y3_test,d_pred)))

#일 DT 성능평가
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(max_depth=5
                           ,min_samples_leaf=25
                            ,random_state=0)
tree.fit(X3_train_selected,y3_train)
pred=tree.predict(X3_test_selected)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y3_test,pred))
print('MSE:',mean_squared_error(y3_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y3_test,pred)))
print('R^2:',r2_score(y3_test,pred))

#일 Random forest 성능평가
from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=28,max_depth=14,random_state=0)
forest.fit(X3_train_selected,y3_train)
pred=forest.predict(X3_test_selected)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y3_test,pred))
print('MSE:',mean_squared_error(y3_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y3_test,pred)))
print('R^2:',r2_score(y3_test,pred))



#일 LR k-fold CV
from sklearn.model_selection import cross_val_score
lr=LinearRegression()
lr_cv_scores=cross_val_score(lr,X3,y3,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(lr_cv_scores))


# In[180]:


#일 KNN k-fold CV
from sklearn.model_selection import cross_val_score
knn=KNeighborsRegressor(n_neighbors=12,weights='uniform')
knn_cv_scores=cross_val_score(knn,X3,y3,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(knn_cv_scores))



# 일 DT k-fold CV
from sklearn.model_selection import cross_val_score
tree=DecisionTreeRegressor(max_depth=5
                           ,min_samples_leaf=25
                            ,random_state=0)
tree_cv_scores=cross_val_score(tree,X3,y3,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(tree_cv_scores))


#일 Random forest k-fold CV
from sklearn.model_selection import cross_val_score
forest=RandomForestRegressor(n_estimators=28,max_depth=14,random_state=0)
forest_cv_scores=cross_val_score(forest,X3,y3,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(forest_cv_scores))






# In[40]:


#순위 뽑아내기**중요(금토일 가장 잘된 모델 하나씩만 할것)

rank_df=pd.read_excel('MAYtotal_standard_둘째주만.xlsx')
rank_df.columns
rank_org=rank_df[['station','weekday_mean']]
#둘째주만 데이터는 둘째주만 잇고, weekdaymean이잇음.거기서 위 두개만 가져와서 형성

rank_sun=np.array(rank_df[['N_weekday_mean', 'N_foodindustry', 'N_buss_density', 'N_univ_num',
       'N_trend_thu', 'N_trend_fri', 'N_trend_sat', 'N_trend_sun',
       'N_pop_fri']]) #변수선택으로 줄여진 변수만
rank=pd.DataFrame(rank_org)

#일요일은 랜덤포레스트 선택
from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=28,max_depth=14,random_state=0)
forest.fit(X3_train_selected,y3_train)
result=pd.DataFrame(forest.predict(rank_sun)) #forest,X3-y3, test_sun만 바꾸면 됨

#데이터프레임형성
rank['predicted_sun']=result


# In[41]:


#정규화된 예측데이터프레임을 역순서대로 정렬해서 weekdaymean이랑 pop량 붙임
rank=rank.sort_values(['station'],ascending=True)
rank=pd.concat([rank,rank_df.iloc[:,-1:]],axis=1)

#성공 ㅠ 넘복잡해따ㅠ
rank.head(40)


# In[42]:


#증감율 계산 sun만 요일 바꿔주면 댐
rank['change_rate_sun']=(rank['predicted_sun']-rank['weekday_mean'])/rank['weekday_mean']
rank=rank.sort_values(['change_rate_sun'], ascending=True)
rank.iloc[0:100,:].to_excel('둘째주_일예측_그래프화필요.xls')

