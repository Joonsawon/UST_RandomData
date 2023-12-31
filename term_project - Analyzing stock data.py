# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 07:20:26 2023

@author: ADMIN
"""
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.offline import plot

#%%

# 2023년도 거래량 상위 10개중 기준가 가장 낮은 "갤럭시아머니트리" 
# http://data.krx.co.kr/contents/MMC/RANK/rank/MMCRANK006.cmd

ticker = yf.Ticker('094480.KQ')
 ticker.history
df_minute = ticker.history( interval='1m',
               start='2023-10-20',
               end='2023-10-27',
               actions=True, auto_adjust=True)

df_minute


fig = px.line(df_minute['Open'], markers=False, title ='위지트 시가 데이터')
fig.update_xaxes(title ='Wavelength / mm')
fig.update_yaxes(title = 'Intensity / A.U.')
plot(fig)



## 0. 11월 한달치 데이터 모으기, 단 로우는 하루 치임 ()

#%%
#반복문으로 구해야 하는데, 장이 안열리는 날에는 어떻게 나오는지 확인하자

df_minute = ticker.history( interval='1m',
               start='2023-11-28',
               end='2023-11-28',
               actions=True, auto_adjust=True)

df_minute

# 빈 데이터 프레임이 나오고 인덱스로 불러오면 에러가 난다.
# if문으로 값이 없으면 넘어가게 하자. 
# df_minute.shape[0] 으로 행의 갯수를 구할 수 있다. 
# 1분간격 데이터는  지난 30일로 제한되어있음..

stock_data = [] # 숫자를 담을 리스트

df_minute_3 = ticker.history( interval='1m',
               start='2023-11-28',
               end='2023-12-05',
               actions=True, auto_adjust=True)

df_minute

#짜피 한달이니까 하드 코딩하자...

## 11월 7일부터 12월 4일까지 데이터를 df_minute_0~3 에 담았다. 
#합치기 후
concat = pd.concat([df_minute_0,df_minute_1,df_minute_2,df_minute_3])

# 일단 CSV로 저장.. (귀중한 녀석)
concat.to_csv('갤럭시아머니트리.csv')


##1. 종가가 시가보다 이상일 때의 하루 데이터 모음

rise = concat[concat['Open'] < concat['Close']]

## 아... 일단 day 데이터를 가져와서 해야겠다

df_day = ticker.history( interval='1d',
               start='2023-19-07',
               end='2023-12-05',
               actions=True, auto_adjust=True)

rise = df_day[df_day['Open'] < df_day['Close']]

#%%  rise한 날만 선택하기

# 특정 날짜 데이터 가져오기

a = concat[pd.DatetimeIndex(concat.index).day == 29]  #11월 7일 데이터

# 일단 하루 데이터의 FFT를 해보고, 그 의미를 분석해보자. 

fig = px.line(a['Open'], markers=False, title ='위지트 시가 데이터(11/29)')
fig.update_xaxes(title ='time / min')
fig.update_yaxes(title = 'Open / ₩')
plot(fig)

a_fft = np.fft.fft(a['Open'])
a_fft_M = abs(a_fft)*2/357
a_fft_phase = np.angle(a_fft)/np.pi

fig = px.line(a_fft_M[1:], markers=False, title ='위지트 시가 데이터(11/29)')
fig.update_xaxes(title ='time / min')
fig.update_yaxes(title = 'Open / ₩')
plot(fig)

#%%
# DC 성분이 너무 큼.. offset을 제거해주자.
a_fft = np.fft.fft(a['Open']-a['Open'].min())
a_fft_M = abs(a_fft)*2/357


freq = np.fft.fftfreq(a_fft.shape[0], d=60)



fig = px.bar(x=freq[1:357//2],y=a_fft_M[1:357//2], title ='위지트 시가 데이터(11/29)')
fig.update_xaxes(title ='frequency / Hz')
fig.update_yaxes(title = 'Magnitude')
plot(fig)


#%% 내려 갈 때의 주기 정보


## 하루라는 시간동안 주가가 변하는 주기


 fall = df_day[df_day['Open'] > df_day['Close']]

b = concat[pd.DatetimeIndex(concat.index).day == 15]  #11월 7일 데이터
# 11/15 거래량    3231984
# 11/29 거래량 30400010
# 일단 하루 데이터의 FFT를 해보고, 그 의미를 분석해보자.

fig = px.line(b['Open'], markers=False, title ='위지트 시가 데이터(11/29)')
fig.update_xaxes(title ='time / min')
fig.update_yaxes(title = 'Open / ₩')
plot(fig)


b_fft = np.fft.fft(b['Open'])
b_fft_M = abs(b_fft)*2/357
b_fft_phase = np.angle(b_fft)/np.pi

fig = px.bar(x=freq[1:357//2],y=b_fft_M[1:357//2], title ='하락')
fig.update_xaxes(title ='frequency / Hz')
fig.update_yaxes(title = 'Magnitude')
plot(fig)

fig = px.bar(x=freq[1:357//2],y=b_fft_phase[1:357//2], title ='하락')
fig.update_xaxes(title ='frequency / Hz')
fig.update_yaxes(title = 'Phase / pi')
plot(fig)

fig = px.bar(x=freq[1:357//2],y=a_fft_phase[1:357//2], title ='상승')
fig.update_xaxes(title ='frequency / Hz')
fig.update_yaxes(title = 'Phase / pi')
plot(fig)

fig = px.bar(x=freq[1:357//2],y=a_fft_M[1:357//2], title ='상승')
fig.update_xaxes(title ='frequency / Hz')
fig.update_yaxes(title = 'Magnitude')
plot(fig)

#%% # 4. 그래프 확인

df = pd.DataFrame({'상승': a_fft_M[1:357//2], '하락':b_fft_M[1:357//2]})
df.index = freq[1:357//2] # 파장 설정
fig = px.bar(df,  title ='상승장 하락장 비교',barmode="group")
fig.update_xaxes(title ='frequency / Hz')
fig.update_yaxes(title = 'Magnitude')
#fig.add_bar(x=data_wl, y= data_interf, name='name')
plot(fig)