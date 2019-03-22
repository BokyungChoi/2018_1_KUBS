# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 17:21:15 2018

@author: ____
"""

import os
import numpy as np
import pandas as pd

weather_df=pd.read_excel("weather_data.xls")
weather_df.shape

#데이터를 각 주마다 분리하기 위해 iloc 함수로 주마다 분리
week1_weather_df=weather_df.iloc[:69]
week1_weather_df.tail()
week1_weather_df.to_csv("week1_weather.txt")

week2_weather_df=weather_df.iloc[69:138]
week2_weather_df.tail()
week2_weather_df.to_csv("week2_weather.txt")

week3_weather_df=weather_df.iloc[138:207]
week3_weather_df.tail()
week3_weather_df.to_csv("week3_weather.txt")

week4_weather_df=weather_df.iloc[207:]
week4_weather_df.tail()
week3_weather_df.to_csv("week4_weather.txt")
