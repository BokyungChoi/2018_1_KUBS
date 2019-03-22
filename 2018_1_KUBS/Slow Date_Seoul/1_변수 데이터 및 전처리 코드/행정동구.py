# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 00:58:23 2018

@author: BDI
"""

import os
os.chdir(r'C:\Users\BDI\Documents\3-1\데이터관리와 지적경영')
import pandas as pd

#1~4호선
stat_df = pd.read_csv('subway.csv')
stat_df.head()
f1 = lambda x: x.split()[1] #구주소 띄어쓰기로 구분해서 행정구 컬럼 만들기
stat_df['행정구'] = stat_df['구주소'].map(f1)
stat_df.head()
f2 = lambda x: x.split()[2] if len(x.split()) >= 3 else 0
stat_df['행정동'] = stat_df['구주소'].map(f2) #구주소 띄어쓰기로 구분해서 행정동 컬럼 만들기
stat_df.head()
stat_df2 = stat_df.drop(['구주소', '도로명주소', '전화번호'], axis = 1) #관련없는 변수 삭제
stat_df2.head()
stat_df2.to_csv("14.csv", index = False)

#5~8호선
sub_df = pd.read_csv('subb.csv')
sub_df.head()
sub_df['행정구'] = sub_df['지번주소'].map(f1) #지번주소 띄어쓰기로 구분해서 행정구 컬럼 만들기
sub_df.head()
sub_df['행정동'] = sub_df['지번주소'].map(f2) #지번주소 띄어쓰기로 구분해서 행정동 컬럼 만들기
sub_df.head()
sub_df2 = sub_df.drop(['연번', '도로명주소', '지번주소', '우편번호', '전화번호'], axis = 1) #관련없는 변수 삭제
sub_df2.head()
sub_df2.to_csv("58.csv", index = False)

#1~8호선 합치기
hi_df = pd.concat([stat_df2, sub_df2], axis = 0)
hi_df.reset_index()
hi_df.head()
hi_df.to_csv("add.txt", index = False)

sam_df = pd.read_csv('add.txt')
sam_df.head()