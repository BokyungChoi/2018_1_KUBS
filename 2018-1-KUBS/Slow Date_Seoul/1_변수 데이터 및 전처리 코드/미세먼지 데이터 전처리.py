# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 17:09:32 2018

@author: 김솔이
"""

import os
import numpy as np
import pandas as pd

os.chdir(r'C:\Users\김솔이\Desktop\데이터관리 Team Project')
dust_df=pd.read_excel("dust_data.xls")

dust_df.head()

# 각 변수의 min/max 확인
dust_df.describe()

# 불필요한 column 제거(iloc함수)
#미세먼지, 초미세먼지 데이터만 사용
dust_df1=dust_df.iloc[:,:4]
dust_df1.head()

dust_df1.to_csv("dust_edit.txt")