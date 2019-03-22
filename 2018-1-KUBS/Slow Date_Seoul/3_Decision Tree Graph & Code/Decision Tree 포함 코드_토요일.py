
# coding: utf-8

# In[2]:


import os 
import pandas as pd
import numpy as np
import sklearn


# In[3]:


os.chdir(r"C:\Users\김솔이\Desktop\데이터관리 Team Project\전처리 데이터+코드")


# In[295]:


mays_df=pd.read_excel('MAYtotal_Standard_new.xls')


# In[212]:


mays_df.columns


# In[159]:


mays_df.head()


# In[296]:


#토요일예측 ndarray로 변경하기
X2=np.array(mays_df[['N_weekday_mean', 'N_fri_temp', 'N_fri_rain',
       'N_sat_temp', 'N_sat_rain', 'N_fri_미세',
       'N_fri_초미세', 'N_sat_미세', 'N_sat_초미세',
       'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri', 'N_trend_sat','N_pop_fri']])
y2=np.array(mays_df['pop_sat'])
print(X2[:3],y2[:3])


# In[214]:


#토요일 Data split
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test=train_test_split(X2,y2,test_size=0.3,random_state=0)

print(X2_train.shape)
print(y2_train.shape)
print(X2_test.shape)
print(y2_test.shape)


# In[215]:


#토요일 feature selection(이걸로 선택!)-일변량통계
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

select=SelectPercentile(percentile=60,score_func=f_regression)
select.fit(X2_train,y2_train)

#선택된 변수 확인
may2_df=mays_df[['N_weekday_mean', 'N_fri_temp', 'N_fri_rain',
       'N_sat_temp', 'N_sat_rain', 'N_fri_미세',
       'N_fri_초미세', 'N_sat_미세', 'N_sat_초미세',
       'N_foodindustry', 'N_population_density', 'N_buss_density',
       'N_univ_num', 'N_trend_thu', 'N_trend_fri', 'N_trend_sat','N_pop_fri']]
may2_df.columns[np.where(select.get_support()==True)]


# In[216]:


#transform을 이용해 선택된 변수만 선택
X2_train_selected=select.transform(X2_train)
X2_train_selected.shape
X2_test_selected=select.transform(X2_test)
X2_test_selected.shape


# In[217]:


#토요일 Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X2_train_selected,y2_train)
print('Training R^2:', lr.score(X2_train_selected,y2_train))
print('Test R^2:', lr.score(X2_test_selected,y2_test))


# In[218]:


#토요일 Coefficient & intercept
m_names=['N_weekday_mean', 'N_fri_rain', 'N_sat_temp', 'N_foodindustry',
       'N_buss_density', 'N_univ_num', 'N_trend_thu', 'N_trend_fri',
       'N_trend_sat', 'N_pop_fri']
m_coef=lr.coef_
c_i=pd.Series(m_coef, index=m_names)
c_i=c_i.sort_values(ascending=False)
print(c_i)


# In[220]:


##그래프 생성
b_df=pd.DataFrame(c_i)
b_df.to_csv('sat_coef.csv',index=True)
#그래프 첨부


# In[221]:


#토요일 KNN-for (n_neighbors=2가 가장 높은 accuracy)
from sklearn.neighbors import KNeighborsRegressor
for i in list(range(1,100)):
    knn=KNeighborsRegressor(n_neighbors=i,weights='uniform')
    knn.fit(X2_train_selected,y2_train)
    print('KNN-',i)
    print('Training Accuracy:', knn.score(X2_train_selected,y2_train))
    print('Test Accuracy:', knn.score(X2_test_selected,y2_test))
    print('\n')


# In[222]:


#토요일 2-NN
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=2,weights='uniform')
knn.fit(X2_train_selected,y2_train)
    
print('Training Accuracy:', knn.score(X2_train_selected,y2_train))
print('Test Accuracy:', knn.score(X2_test_selected,y2_test))


# In[251]:


#토요일 Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(max_depth=7,min_samples_split=5, min_samples_leaf=2,random_state=0)
tree.fit(X2_train_selected,y2_train)
print('Training R^2:', tree.score(X2_train_selected,y2_train))
print('Test R^2:', tree.score(X2_test_selected,y2_test))

#Decision Tree image
from sklearn.tree import export_graphviz
dot_data=export_graphviz(tree, out_file=None, impurity=False, filled=True)

from IPython.display import Image
import pydotplus

graph=pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# In[265]:


#토요일 RandomForest
from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=28,max_depth=12,random_state=0)
forest.fit(X2_train_selected,y2_train)
print('Training R^2:', forest.score(X2_train_selected,y2_train))
print('Test R^2::', forest.score(X2_test_selected,y2_test))


# In[266]:


f_importances=forest.feature_importances_

for name, importance in zip(m_names,f_importances):
    print(name, importance)


# In[267]:


#sort
s_importances=pd.Series(f_importances,index=m_names)
s_importances.sort_values(ascending=False)


# In[268]:


#토 LR 성능평가
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X2_train_selected,y2_train)
pred=lr.predict(X2_test_selected)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y2_test,pred))
print('MSE:',mean_squared_error(y2_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y2_test,pred)))
print('R^2:',r2_score(y2_test,pred))


# In[269]:


#토 dummy 성능평가
from sklearn.dummy import DummyRegressor
dummy=DummyRegressor(strategy='mean')
dummy.fit(X2_train_selected,y2_train)
d_pred=dummy.predict(X2_test_selected)
print('\n-DummyRegressor')
print('MAE:',mean_absolute_error(y2_test,d_pred))
print('MSE:',mean_squared_error(y2_test,d_pred))
print('RMSE:',np.sqrt(mean_squared_error(y2_test,d_pred)))


# In[270]:


#토 KNN 성능평가
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=2,weights='uniform')
knn.fit(X2_train_selected,y2_train)
pred=knn.predict(X2_test_selected)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y2_test,pred))
print('MSE:',mean_squared_error(y2_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y2_test,pred)))
print('R^2:',r2_score(y2_test,pred))


# In[272]:


#토 DT 성능평가
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(max_depth=7,min_samples_split=5, min_samples_leaf=2
                            ,random_state=0)
tree.fit(X2_train_selected,y2_train)
pred=tree.predict(X2_test_selected)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y2_test,pred))
print('MSE:',mean_squared_error(y2_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y2_test,pred)))
print('R^2:',r2_score(y2_test,pred))


# In[273]:


#토 RF 성능평가
from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=28,max_depth=12,random_state=0)
forest.fit(X2_train_selected,y2_train)
pred=forest.predict(X2_test_selected)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y2_test,pred))
print('MSE:',mean_squared_error(y2_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y2_test,pred)))
print('R^2:',r2_score(y2_test,pred))


# In[297]:


#토 LR k-fold CV
from sklearn.model_selection import cross_val_score
lr=LinearRegression()
lr_cv_scores=cross_val_score(lr,X2,y2,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(lr_cv_scores))


# In[298]:


#토 KNN k-fold CV
from sklearn.model_selection import cross_val_score
knn=KNeighborsRegressor(n_neighbors=12,weights='uniform')
knn_cv_scores=cross_val_score(knn,X2,y2,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(knn_cv_scores))


# In[299]:


#토 DT k-fold CV
from sklearn.model_selection import cross_val_score
tree=DecisionTreeRegressor(max_depth=6
                           ,min_samples_leaf=30
                            ,random_state=0)
tree_cv_scores=cross_val_score(tree,X2,y2,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(tree_cv_scores))


# In[300]:


#토 RF k-fold CV
from sklearn.model_selection import cross_val_score
forest=RandomForestRegressor(n_estimators=16,max_depth=11,random_state=0)
forest_cv_scores=cross_val_score(forest,X2,y2,cv=5,scoring='explained_variance')
print('explained_variance:',np.mean(forest_cv_scores))


# In[301]:


#순위 뽑아내기**중요(금토일 가장 잘된 모델 하나씩만 할것)

rank_df=pd.read_excel('MAYtotal_standard_둘째주만.xlsx')
rank_df.columns
rank_org=rank_df[['station','weekday_mean']]
#둘째주만 데이터는 둘째주만 잇고, weekdaymean이잇음.거기서 위 두개만 가져와서 형성

rank_sat=np.array(rank_df[['N_weekday_mean', 'N_fri_rain', 'N_sat_temp', 'N_foodindustry',
       'N_buss_density', 'N_univ_num', 'N_trend_thu', 'N_trend_fri',
       'N_trend_sat', 'N_pop_fri']]) #변수선택으로 줄여진 변수만
rank=pd.DataFrame(rank_org)

#토요일은 RF 선택
from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=28,max_depth=12,random_state=0)
forest.fit(X2_train_selected,y2_train)
result=pd.DataFrame(forest.predict(rank_sat)) #forest,X3-y3, test_sun만 바꾸면 됨

#데이터프레임형성
rank['predicted_sat']=result


# In[302]:


#정규화된 예측데이터프레임을 역순서대로 정렬해서 weekdaymean이랑 pop량 붙임
rank=rank.sort_values(['station'],ascending=True)
rank=pd.concat([rank,rank_df.iloc[:,-2:-1]],axis=1)

#성공 
rank.head(10)


# In[303]:


#증감율 계산 sun만 요일 바꿔주면 댐
rank['change_rate_sat']=(rank['predicted_sat']-rank['weekday_mean'])/rank['weekday_mean']
rank=rank.sort_values(['change_rate_sat'], ascending=True)
rank.iloc[0:100,:].to_excel('둘째주_토예측_그래프화필요.xls')


# In[ ]:


#끝~~

