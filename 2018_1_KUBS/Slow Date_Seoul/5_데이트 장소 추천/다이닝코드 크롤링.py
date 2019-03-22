
# coding: utf-8

# In[45]:


'''
Dining Code 키워드별 1위-10위 맛집 크롤링

'''

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
dining_code('한양대')


# In[47]:


#for로 리스트 내 맛집 데이터 추출
dining_rank=pd.DataFrame()
for i in station_list:
    dining_rank=dining_rank.append(pd.Series(dining_code(i)),ignore_index=True)


# In[5]:


import os
import pandas as pd
import numpy as np
import sklearn
os.chdir(r'C:\Users\최보경\Desktop\2018JUNIOR\01-데이터관리와지적경영\teamdata\final')


# In[9]:


#리스트 읽어오기
may_df=pd.read_excel('MAY1(솔이님+트렌드+유동량변화율로정리).xls')
station_list=list(may_df['station'].unique())


# In[40]:


x=pd.DataFrame()
x=x.append(pd.Series(['d', 'e','f']),ignore_index=True )
x=x.append(pd.Series(['d', 'e','f']),ignore_index=True )
x


# In[49]:


dining_rank.columns=['1위','2위','3위','4위','5위','6위','7위','8위','9위','10위']


# In[51]:


dining_rank.to_excel('Dining_rank.xls')

