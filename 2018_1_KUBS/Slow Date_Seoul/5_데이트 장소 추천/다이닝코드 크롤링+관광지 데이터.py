
# coding: utf-8

# In[48]:


import os
import pandas as pd
import numpy as np


# In[53]:


place_df=pd.read_excel('서울교통공사 역출구별 관광지 정보.xls')


# In[55]:


diningcode_df=pd.read_excel('Dining_rank.xls')


# In[56]:


#다이닝코드 크롤링 데이터랑 서울교통공사 역출구별 관광지를 합침! 역이름에 맞춰서
Slowdate_recommendation=pd.merge(diningcode_df,place_df,how='left',left_on='station',right_on='역명')


# In[64]:


Slowdate_recommendation.to_excel('Slowdate_recommendation.xls')


# In[65]:


#상위랭킹의 역의 맛집과 관광지를 바로 보여주는 코드
slow_df=Slowdate_recommendation.set_index(['station'])
slow_df.drop('역명',axis=1,inplace=True)
pd.DataFrame(slow_df.loc['역삼'])


# In[58]:


#위 코드 복잡해보여서 또 다른 시도
dining_df = diningcode_df.set_index(['station'])
#place_df = place_df.set_index(['역명'])
# place_df.drop(['호선',''])
a=pd.DataFrame(dining_df.loc['방배'])
b=pd.DataFrame(place_df.loc['방배'])
print(a,b)


# In[9]:


date_rec=pd.read_excel('Slowdate_recommendation.xls')
# date_rec.drop('역명',axis=1,inplace=True)

