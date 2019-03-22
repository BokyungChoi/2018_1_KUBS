# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:43:37 2018

@author: BDI
"""

import pandas as pd

#사업체종사자수/면적 변수 구하기 위해 사업체현황.xls파일, 동면적.xls파일 정리

buss_df = pd.read_excel('사업체현황.xls')
buss_df2 = buss_df.drop(['기간', '자치구', '합계.2', '농업 임업 및 어업', '농업 임업 및 어업.1', '광업', '광업.1', '제조업', '제조업.1', '건설업', '건설업.1', '전기 가스 증기 및 수도사업', '전기 가스 증기 및 수도사업.1', '하수 · 폐기물 처리 원료재생 및 환경복원업', '하수 · 폐기물 처리 원료재생 및 환경복원업.1', '도매 및 소매업', '도매 및 소매업.1', '운수업', '운수업.1', '숙박 및 음식점업.1', '출판 영상 방송통신 및 정보서비스업', '출판 영상 방송통신 및 정보서비스업.1', '금융 및 보험업', '금융 및 보험업.1', '부동산업 및 임대업', '부동산업 및 임대업.1', '전문 과학 및 기술 서비스업', '전문 과학 및 기술 서비스업.1', '사업시설관리 및 사업지원 서비스업', '사업시설관리 및 사업지원 서비스업.1', '공공행정 국방 및 사회보장 행정', '공공행정 국방 및 사회보장 행정.1', '교육 서비스업', '교육 서비스업.1', '보건업 및 사회복지 서비스업', '보건업 및 사회복지 서비스업.1', '예술 스포츠 및 여가관련 서비스업', '예술 스포츠 및 여가관련 서비스업.1', '협회 및 단체 수리  및 기타 개인 서비스업', '협회 및 단체 수리  및 기타 개인 서비스업.1'], axis=1) #관련없는 변수 삭제
buss_df2.head()
buss_df3 = buss_df2.drop([0], axis=0)
buss_df3.head()
buss_df3.columns = ['dong', 'sum', 'sump', 'eat']
buss_df3.head()

type(buss_df3["sump"])
buss_df3["sump"] = buss_df3.sump.astype(float)
buss_df3["eat"] = buss_df3.eat.astype(float)
buss_df3["diff"] = buss_df3["sump"].subtract(buss_df3["eat"], fill_value=0)
buss_df3.head()
buss_df4 = buss_df3.drop(['sum', 'sump', 'eat'], axis=1)
buss_df4.head()
buss_df4.to_csv("buss_df4.csv", index=False)
bussdict = buss_df4.set_index('dong').to_dict()
print(bussdict)

#동별면적
area_df = pd.read_excel('동면적.xls')
area_df.head()
area_df2 = area_df.drop([0, 1], axis=0)
area_df2.head()
area_df3 = area_df.drop(['기간', '자치구', '면적.1', '동.1', '동.2', '통', '반'], axis=1) #관련없는 변수삭제
area_df3.head()
area_df3.to_csv("area_df3.csv", index=False)
area_df4 = area_df3.drop([0, 1, 2], axis=0) #관련없는 변수삭제
area_df5 = area_df4.reset_index().head()
area_df5.head()
area_df6 = area_df5.drop(['index'],axis=1)
area_df6.head()
area_df6.columns = ['dong', 'area']
type(area_df6['area'])
area_df6['area'] = area_df6.area.astype(float)
f_area = lambda x: x*1000 #변수값이 너무 작아보여서 1km제곱기준이어서 1000곱함
area_df6['area0'] = area_df6['area'].map(f_area)
area_df6.head()
area_df6.to_csv("area_df6.csv", index=False)

area_df7 = area_df6.drop(['area'], axis=1)
area_df7.head()
area_df7['area0'] = area_df7.area0.astype(float)

#사업체종사자수/면적 나누기 하려했으나 안돼서 결국 엑셀로 함
buss_df4["ratio"] = buss_df4["diff"].divide(area_df7["area0"], fill_value=0)
buss_df4.head()
buss_df5 = buss_df4.drop(['diff'], axis=1)
buss_df5.head()
buss_df6 = buss_df5.reset_index()
buss_df6.head()
buss_df6 = buss_df6.drop(['area'], axis=1)
buss_df6.head()
buss_df6.to_csv("buss_df6.csv", index=False)

buss_df5['area'] = area_df6['area']
buss_df5.head()

areadict = buss_df5.set_index('dong').to_dict()
print(areadict)

#엑셀로 나누기 연산한 뒤 불러와서 dictionary로 만듦
busar_df = pd.read_excel('사업체 면적.xlsx')
busar_df.head()
busardict = busar_df.set_index('dong').to_dict()
print(busardict)