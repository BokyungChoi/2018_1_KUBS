
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import sklearn


# In[2]:


os.getcwd()


# In[3]:


os.chdir(r"C:\\Users\\최보경\\Desktop\\2018JUNIOR\\01-데이터관리와지적경영\\teamdata\\trend")


# In[4]:


os.getcwd()


# In[21]:


os.chdir(r"C:\\Users\\최보경\\Desktop\\2018JUNIOR\\01-데이터관리와지적경영\\teamdata")

may1_df=pd.read_csv('may1.txt')
may1_df.head()

dong_df=pd.read_csv('REALdong.txt')
dong_df=dong_df[['호선','행정구','역명','행정동']]
dong_df.head()


# In[301]:


os.getcwd()


# In[332]:


os.getcwd()
os.chdir(r"C:\\Users\\최보경\\Desktop\\2018JUNIOR\\01-데이터관리와지적경영\\teamdata\\trend")


# In[330]:


#읽어오기 코드
may1=pd.read_excel('MAY1.xls')
may2=pd.read_excel('MAY2.xls')
may3=pd.read_excel('MAY3.xls')
may4=pd.read_excel('MAY4.xls')


# In[333]:


#내보내기 코드
may1.to_excel('MAY1.xls')
may2.to_excel('MAY2.xls')
may3.to_excel('MAY3.xls')
may4.to_excel('MAY4.xls')


# In[304]:


station_df=pd.read_excel('MAY1.xls')


# In[305]:


station_list=list(station_df['station'].unique())


# In[315]:


station_df=station_df.sort_values(by='station', ascending=True)
station_df.tail()
station_df=station_df.reset_index()


# In[331]:


#reset index

may4=may4.sort_values(by='station', ascending=True)
may4=may4.reset_index()
may4.head()



# In[336]:


#역 이름 리스트화
station_list=list(station_df['station'].unique())
may1.shape


# In[353]:


station_list


# In[389]:


x1=pd.DataFrame()
#week1목금토일 트렌드 읽어서 붙이기
for i in station_list:
    y=pd.read_excel('%s.xlsx'%i)
    y=y.iloc[10:14,2:4]
    y=y.T
    y.columns=['','','','']
    x1=pd.concat([x1,y],axis=0,ignore_index=True)
    x1=x1.reset_index(drop=True)
    x1.columns = ['','','','']
    del y
    continue
x1.to_excel('trend_week1.xls')


# In[375]:


x3=pd.DataFrame()
#week3목금토일
for i in station_list:
    y=pd.read_excel('%s.xlsx'%i)
    y=y.iloc[21:25,2:4]
    y=y.T
    y.columns=['','','','']
    x3=pd.concat([x3,y],axis=0,ignore_index=True)
    x3=x3.reset_index(drop=True)
    x3.columns = ['','','','']
    del y
    continue
x3.to_excel('trend_week2.xls')


# In[377]:


x4=pd.DataFrame()
#week4목금토일
for i in station_list:
    y=pd.read_excel('%s.xlsx'%i)
    y=y.iloc[31:35,2:4]
    y=y.T
    y.columns=['','','','']
    x4=pd.concat([x4,y],axis=0,ignore_index=True)
    x4=x4.reset_index(drop=True)
    x4.columns = ['','','','']
    del y
    continue
x4.to_excel('trend_week3.xls')


# In[376]:


x5=pd.DataFrame()
#week5목금토일
for i in station_list:
    y=pd.read_excel('%s.xlsx'%i)
    y=y.iloc[38:42,2:4]
    y=y.T
    y.columns=['','','','']
    x5=pd.concat([x5,y],axis=0,ignore_index=True)
    x5=x5.reset_index(drop=True)
    x5.columns = ['','','','']
    del y
    continue
x5.to_excel('trend_week4.xls')


# In[ ]:


#week별로 목금토일 트렌드 데이터 뽑아낸 x1-5를 두고, 각각 merge.


# In[222]:


y=a.iloc[10:14,2:4]
y=y.T


# In[378]:


os.chdir(r"C:\Users\최보경\Desktop\2018JUNIOR\01-데이터관리와지적경영\teamdata\final")


# In[92]:


#읽어오기 코드
may1=pd.read_excel('MAY1_final.xlsx',header=None,Index=None)
may2=pd.read_excel('MAY2_final.xlsx',header=None,Index=None)
may3=pd.read_excel('MAY3_final.xlsx',header=None,Index=None)
may4=pd.read_excel('MAY4_final.xlsx',header=None,Index=None)


# In[54]:


#읽어오기 코드
week1=pd.read_excel('trend_week1.xls',header=None)
week2=pd.read_excel('trend_week2.xls',header=None)
week3=pd.read_excel('trend_week3.xls',header=None)
week4=pd.read_excel('trend_week4.xls',header=None)


# In[ ]:


#merge 시도1
MAY1=pd.merge(may1,x1,how='outer',left_on='station',right_on='index')
MAY1.to_excel('MAY1.xls')


# In[23]:


week1['tmp'] = 1
may1['tmp'] = 1


# In[59]:


week1=week1.dropna()
week1.shape


# In[22]:


#merge 시도2
week1['tmp'] = 1
may1['tmp'] = 1

may1 = pd.merge(may1,week1,how='outer',on=['tmp'])
may1 = may1.drop('tmp', axis=1)


# In[399]:


x1.columns=['index','trend_thu','trend_fri','trend_sat','trend_sun']


# In[394]:


x1=pd.read_excel('trend_week1.xls')


# In[7]:


os.chdir(r"C:\\Users\\최보경\\Desktop\\2018JUNIOR\\01-데이터관리와지적경영\\teamdata\\final")


# In[91]:


may1.head()


# In[27]:


#컬럼순서정리
may1=may1[['index', 'tmp',
       'station', '행정구', '행정동', '20180430', '20180501', '20180502', '20180503',
       'weekday_mean', '20180504', '20180505', '20180506', 'week1_fri_temp',
       'week1_fri_rain', 'week1_sat_temp', 'week1_sat_rain', 'week1_sun_temp',
       'week1_sun_rain', 'foodindustry', 'population_density', 'week1_fri_미세',
       'week1_fri_초미세', 'week1_sat_미세', 'week1_sat_초미세', 'week1_sun_미세',
       'week1_sun_초미세', 'univ_num','trend_thu', 'trend_fri', 'trend_sat', 'trend_sun']] 


# In[36]:


may1.shape

