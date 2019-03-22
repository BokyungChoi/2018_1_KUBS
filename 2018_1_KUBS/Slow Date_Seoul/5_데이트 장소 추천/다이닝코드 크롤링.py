
# coding: utf-8

# In[45]:


'''
Dining Code 키워드별 1위-10위 맛집 크롤링

'''

import os
import pandas as pd
import numpy as np

#함수 코딩
import requests as req
from bs4 import BeautifulSoup

def dining_code(q):
    html=req.get('https://www.diningcode.com/list.php?query=%s'%q).text
    soup=BeautifulSoup(html,'html.parser')
    rank_list=[]
    for idx,tag in enumerate(soup.select(
        "[class~=dc-restaurant-name]"),0):
        name='{}'.format(tag.text)
        rank_list.append(name.replace('\n',''))
    
    del rank_list[0]
    return rank_list


# In[46]:


#함수 활용
dining_code('역 이름')


# In[47]:


#for로 리스트 내 맛집 데이터 추출 (___은 검색어들 리스트 타입으로 넣어주세요)
dining_rank=pd.DataFrame()
for i in _____:
    dining_rank=dining_rank.append(pd.Series(dining_code(i)),ignore_index=True)


# In[49]:


dining_rank.columns=['1위','2위','3위','4위','5위','6위','7위','8위','9위','10위']


# In[51]:


dining_rank.to_excel('Dining_rank.xls')

