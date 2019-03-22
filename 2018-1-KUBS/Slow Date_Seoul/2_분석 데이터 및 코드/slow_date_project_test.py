
# coding: utf-8

# In[1]:


import os 
import pandas as pd
import numpy as np
import sklearn


# In[2]:


os.chdir(r"C:\Users\김솔이\Desktop\데이터관리 Team Project")


# In[3]:


may_df=pd.read_excel('MAYtotal_유동량.xls')


# In[4]:


may_df.columns


# In[21]:


#금요일예측 ndarray로 변경하기
X1=np.array(may_df[['weekday_mean','fri_temp', 'fri_rain', 'fri_미세', 'fri_초미세', 'foodindustry', 'population_density', 'N_busden', 'univ_num', 'trend_thu', 'trend_fri']])
y1=np.array(may_df['pop_fri'])
print(X1[:3],y1[:3])


# In[30]:


#금요일 Data split
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test=train_test_split(X1,y1,test_size=0.3,random_state=0)

print(X1_train.shape)
print(y1_train.shape)
print(X1_test.shape)
print(y1_test.shape)


# In[31]:


#금요일 Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X1_train,y1_train)
print('Training R^2:', lr.score(X1_train,y1_train))
print('Test R^2:', lr.score(X1_test,y1_test))


# In[32]:


#금요일 Coefficient & intercept
m_names=['weekday_mean','fri_temp', 'fri_rain', 'fri_미세', 'fri_초미세', 'foodindustry', 'population_density', 'N_busden', 'univ_num', 'trend_thu', 'trend_fri']
m_coef=lr.coef_
c_i=pd.Series(m_coef, index=m_names)
c_i=c_i.sort_values(ascending=False)
print(c_i)


# In[34]:


##그래프 생성
a_df=pd.DataFrame(c_i)
a_df.to_csv('fri_coef.csv',index=True)
#그래프 첨부


# In[49]:


#금요일 KNN-for (n_neighbors=7이 가장 높은 accuracy)
from sklearn.neighbors import KNeighborsRegressor
for i in list(range(1,100)):
    knn=KNeighborsRegressor(n_neighbors=i,weights='uniform')
    knn.fit(X1_train,y1_train)
    print('KNN-',i)
    print('Training Accuracy:', knn.score(X1_train,y1_train))
    print('Test Accuracy:', knn.score(X1_test,y1_test))
    print('\n')


# In[47]:


#금요일 KNN
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=7,weights='uniform')
knn.fit(X1_train,y1_train)

print('Training R^2:', knn.score(X1_train,y1_train))
print('Test R^2:', knn.score(X1_test,y1_test))


# In[184]:


#토요일예측 ndarray로 변경하기
X2=np.array(may_df[['fri_temp', 'fri_rain', 'fri_미세', 'fri_초미세','sat_temp','sat_rain', 'sat_미세', 'sat_초미세', 'foodindustry', 'population_density', 'N_busden', 'univ_num', 'trend_thu', 'trend_fri','trend_sat','pop_fri']])
y2=np.array(may_df['pop_sat'])
print(X2[:3],y2[:3])


# In[185]:


#토요일 Data split
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test=train_test_split(X2,y2,test_size=0.3,random_state=0)

print(X2_train.shape)
print(y2_train.shape)
print(X2_test.shape)
print(y2_test.shape)


# In[186]:


#토요일 Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X2_train,y2_train)
print('Training R^2:', lr.score(X2_train,y2_train))
print('Test R^2:', lr.score(X2_test,y2_test))


# In[187]:


#토요일 Coefficient & intercept
m_names=['fri_temp', 'fri_rain', 'fri_미세', 'fri_초미세','sat_temp','sat_rain', 'sat_미세', 'sat_초미세', 'foodindustry', 'population_density', 'N_busden', 'univ_num', 'trend_thu', 'trend_fri','trend_sat','pop_fri']
m_coef=lr.coef_
c_i=pd.Series(m_coef, index=m_names)
c_i=c_i.sort_values(ascending=False)
print(c_i)


# In[110]:


##그래프 생성
b_df=pd.DataFrame(c_i)
b_df.to_csv('sat_coef.csv',index=True)
#그래프 첨부


# In[111]:


#토요일 KNN-for (n_neighbors=12가 가장 높은 accuracy)
from sklearn.neighbors import KNeighborsRegressor
for i in list(range(1,100)):
    knn=KNeighborsRegressor(n_neighbors=i,weights='uniform')
    knn.fit(X2_train,y2_train)
    print('KNN-',i)
    print('Training Accuracy:', knn.score(X2_train,y2_train))
    print('Test Accuracy:', knn.score(X2_test,y2_test))
    print('\n')


# In[112]:


#토요일 KNN
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=12,weights='uniform')
knn.fit(X2_train,y2_train)

print('Training R^2:', knn.score(X2_train,y2_train))
print('Test R^2:', knn.score(X2_test,y2_test))


# In[120]:


#일요일예측 ndarray로 변경하기
X3=np.array(may_df[['weekday_mean','sat_temp','sat_rain', 'sat_미세', 'sat_초미세','sun_temp', 'sun_rain', 'sun_미세', 'sun_초미세','foodindustry', 'population_density', 'N_busden', 'univ_num', 'trend_thu', 'trend_fri','trend_sat','trend_sun','pop_fri','pop_sat']])
y3=np.array(may_df['pop_sun'])
print(X3[:3],y3[:3])


# In[121]:


#일요일 Data split
from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test=train_test_split(X3,y3,test_size=0.3,random_state=0)

print(X3_train.shape)
print(y3_train.shape)
print(X3_test.shape)
print(y3_test.shape)


# In[122]:


#일요일 Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X3_train,y3_train)
print('Training R^2:', lr.score(X3_train,y3_train))
print('Test R^2:', lr.score(X3_test,y3_test))


# In[123]:


#일요일 Coefficient & intercept
m_names=['weekday_mean', 'sat_temp','sat_rain', 'sat_미세', 'sat_초미세','sun_temp', 'sun_rain', 'sun_미세', 'sun_초미세','foodindustry', 'population_density', 'N_busden', 'univ_num', 'trend_thu', 'trend_fri','trend_sat','trend_sun','pop_fri','pop_sat']
m_coef=lr.coef_
c_i=pd.Series(m_coef, index=m_names)
c_i=c_i.sort_values(ascending=False)
print(c_i)


# In[124]:


##그래프 생성
c_df=pd.DataFrame(c_i)
c_df.to_csv('sun_coef.csv',index=True)
#그래프 첨부


# In[125]:


#일요일 KNN-for (n_neighbors=12가 가장 높은 accuracy)
from sklearn.neighbors import KNeighborsRegressor
for i in list(range(1,100)):
    knn=KNeighborsRegressor(n_neighbors=i,weights='uniform')
    knn.fit(X3_train,y3_train)
    print('KNN-',i)
    print('Training Accuracy:', knn.score(X3_train,y3_train))
    print('Test Accuracy:', knn.score(X3_test,y3_test))
    print('\n')


# In[126]:


#일요일 KNN
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=12,weights='uniform')
knn.fit(X3_train,y3_train)

print('Training R^2:', knn.score(X3_train,y3_train))
print('Test R^2:', knn.score(X3_test,y3_test))


# In[24]:


#정규화
from sklearn.preprocessing import MinMaxScaler
a=may_df.iloc[:,3:27].values.reshape(-1,1)
minmax_scaler=MinMaxScaler(feature_range=(0,1))
a=minmax_scaler.fit_transform(a)
len(a)

#may1['N_univ_num']=a[:]
#may1.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
#리스트 형태로 넣어주기 위해 변환
minmax_scaler=MinMaxScaler().fit(list(onehotfb_df.iloc[:,1:12].values))
minmax_mat=minmax_scaler.transform(list(onehotfb_df.iloc[:,1:12].values))
minmax_mat
minmaxfb_df=pd.DataFrame(minmax_mat)
minmaxfb_df['Lifetime Post Total Impressions']=onehotfb_df['Lifetime Post Total Impressions']
minmaxfb_df.head()


# In[147]:


#금 LR 성능평가
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

from sklearn.dummy import DummyRegressor
dummy=DummyRegressor(strategy='mean')
dummy.fit(X1_train,y1_train)

d_pred=dummy.predict(X1_test)
print('\n-DummyRegressor')
print('MAE:',mean_absolute_error(y1_test,d_pred))
print('MSE:',mean_squared_error(y1_test,d_pred))
print('RMSE:',np.sqrt(mean_squared_error(y1_test,d_pred)))


# In[149]:


#금 KNN 성능평가
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=7,weights='uniform')
knn.fit(X1_train,y1_train)
pred=knn.predict(X1_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y1_test,pred))
print('MSE:',mean_squared_error(y1_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y1_test,pred)))
print('R^2:',r2_score(y1_test,pred))

from sklearn.dummy import DummyRegressor
dummy=DummyRegressor(strategy='mean')
dummy.fit(X1_train,y1_train)

d_pred=dummy.predict(X1_test)
print('\n-DummyRegressor')
print('MAE:',mean_absolute_error(y1_test,d_pred))
print('MSE:',mean_squared_error(y1_test,d_pred))
print('RMSE:',np.sqrt(mean_squared_error(y1_test,d_pred)))


# In[151]:


#토 LR 성능평가
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X2_train,y2_train)
pred=lr.predict(X2_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y2_test,pred))
print('MSE:',mean_squared_error(y2_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y2_test,pred)))
print('R^2:',r2_score(y2_test,pred))

from sklearn.dummy import DummyRegressor
dummy=DummyRegressor(strategy='mean')
dummy.fit(X2_train,y2_train)

d_pred=dummy.predict(X1_test)
print('\n-DummyRegressor')
print('MAE:',mean_absolute_error(y2_test,d_pred))
print('MSE:',mean_squared_error(y2_test,d_pred))
print('RMSE:',np.sqrt(mean_squared_error(y2_test,d_pred)))


# In[154]:


#토 KNN 성능평가
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=12,weights='uniform')
knn.fit(X2_train,y2_train)
pred=knn.predict(X2_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y2_test,pred))
print('MSE:',mean_squared_error(y2_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y2_test,pred)))
print('R^2:',r2_score(y2_test,pred))

from sklearn.dummy import DummyRegressor
dummy=DummyRegressor(strategy='mean')
dummy.fit(X2_train,y2_train)

d_pred=dummy.predict(X2_test)
print('\n-DummyRegressor')
print('MAE:',mean_absolute_error(y2_test,d_pred))
print('MSE:',mean_squared_error(y2_test,d_pred))
print('RMSE:',np.sqrt(mean_squared_error(y2_test,d_pred)))


# In[155]:


#일 LR 성능평가
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X3_train,y3_train)
pred=lr.predict(X3_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y3_test,pred))
print('MSE:',mean_squared_error(y3_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y3_test,pred)))
print('R^2:',r2_score(y3_test,pred))

from sklearn.dummy import DummyRegressor
dummy=DummyRegressor(strategy='mean')
dummy.fit(X3_train,y3_train)

d_pred=dummy.predict(X3_test)
print('\n-DummyRegressor')
print('MAE:',mean_absolute_error(y3_test,d_pred))
print('MSE:',mean_squared_error(y3_test,d_pred))
print('RMSE:',np.sqrt(mean_squared_error(y3_test,d_pred)))


# In[156]:


#일 KNN 성능평가
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=12,weights='uniform')
knn.fit(X3_train,y3_train)
pred=knn.predict(X3_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('-Error check')
print('MAE:',mean_absolute_error(y3_test,pred))
print('MSE:',mean_squared_error(y3_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y3_test,pred)))
print('R^2:',r2_score(y3_test,pred))

from sklearn.dummy import DummyRegressor
dummy=DummyRegressor(strategy='mean')
dummy.fit(X3_train,y3_train)

d_pred=dummy.predict(X3_test)
print('\n-DummyRegressor')
print('MAE:',mean_absolute_error(y3_test,d_pred))
print('MSE:',mean_squared_error(y3_test,d_pred))
print('RMSE:',np.sqrt(mean_squared_error(y3_test,d_pred)))


# In[175]:


#금 LR k-fold CV
from sklearn.model_selection import cross_val_score
lr=LinearRegression()
lr_cv_scores=cross_val_score(lr,X1,y1,cv=5,scoring='neg_mean_squared_error')
print('Neg_mean_squared_log_error:',np.mean(lr_cv_scores))


# In[176]:


#금 KNN k-fold CV
from sklearn.model_selection import cross_val_score
knn=KNeighborsRegressor(n_neighbors=7,weights='uniform')
knn_cv_scores=cross_val_score(knn,X1,y1,cv=5,scoring='neg_mean_squared_error')
print('Neg_mean_squared_log_error:',np.mean(knn_cv_scores))


# In[177]:


#토 LR k-fold CV
from sklearn.model_selection import cross_val_score
lr=LinearRegression()
lr_cv_scores=cross_val_score(lr,X2,y2,cv=5,scoring='neg_mean_squared_error')
print('Neg_mean_squared_log_error:',np.mean(lr_cv_scores))


# In[178]:


#토 KNN k-fold CV
from sklearn.model_selection import cross_val_score
knn=KNeighborsRegressor(n_neighbors=12,weights='uniform')
knn_cv_scores=cross_val_score(knn,X2,y2,cv=5,scoring='neg_mean_squared_error')
print('Neg_mean_squared_log_error:',np.mean(knn_cv_scores))


# In[179]:


#일 LR k-fold CV
from sklearn.model_selection import cross_val_score
lr=LinearRegression()
lr_cv_scores=cross_val_score(lr,X3,y3,cv=5,scoring='neg_mean_squared_error')
print('Neg_mean_squared_log_error:',np.mean(lr_cv_scores))


# In[180]:


#일 KNN k-fold CV
from sklearn.model_selection import cross_val_score
knn=KNeighborsRegressor(n_neighbors=12,weights='uniform')
knn_cv_scores=cross_val_score(knn,X3,y3,cv=5,scoring='neg_mean_squared_error')
print('Neg_mean_squared_log_error:',np.mean(knn_cv_scores))

