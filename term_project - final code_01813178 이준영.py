# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
import numpy as np
import scipy as sp
from scipy.signal import hilbert
import plotly.express as px
from plotly.offline import plot

#%% 주가 데이터 불러오기

# 2023년도 거래량 상위 10개중 기준가 가장 낮은 "갤럭시아머니트리" 
# http://data.krx.co.kr/contents/MMC/RANK/rank/MMCRANK006.cmd

ticker = yf.Ticker('094480.KQ')
ticker.history
df_minute = ticker.history( interval='1m',
               start='2023-11-07',
               end='2023-11-14',
               actions=True, auto_adjust=True)

# CSV로 저장
df_minute.to_csv('갤럭시아머니트리.csv')

# CSV 불러오기
한달데이터 = pd.read_csv('C:/GitHub/UST_RandomData/갤럭시아머니트리.csv',index_col=0) #인덱스로 사용할 컬럼 설정


#%% Data classification

날짜들 = pd.DatetimeIndex(한달데이터.index).day
저장소 = pd.DataFrame([]) # 날짜별 첫번째 데이터를 담을 리스트 변수 초기화
for i in 날짜들: 
    하루데이터 = 한달데이터[pd.DatetimeIndex(한달데이터.index).day == i] # i번째 날의 데이터 받기
    하루데이터.reset_index(inplace=True,drop=True) # 인덱스 제거
    저장소[f'{i}']=하루데이터.Open[0:300] # Open 데이터 중 300번째 데이터까지 자르기


# 앙상블 평균 계산
mean_value = 저장소.mean(axis='columns') # 열마다의 평균 계산

#그래프
fig = px.line(mean_value, markers=True, title ='장 시간에 따른 주가의 앙상블 평균')
fig.update_xaxes(title ='index')
fig.update_yaxes(title = 'Mean value / ₩')
plot(fig)



#%% 1. FFT 분석


# 상승장 분석 (11/29)

a = 한달데이터[pd.DatetimeIndex(한달데이터.index).day == 29]  #11월 29일 데이터

fig = px.line(a['Open'], markers=False, title ='갤럭시아머니트리 시가 데이터(11/29)')
fig.update_xaxes(title ='time / min')
fig.update_yaxes(title = 'Open / ₩')
plot(fig)

a_fft = np.fft.fft(a['Open'])
a_fft_M = abs(a_fft)*2/357 # normalization

fig = px.line(a_fft_M[1:], markers=False, title ='갤럭시아머니트리 시가 데이터 FFT(11/29)')
fig.update_xaxes(title ='frequency / Hz')
fig.update_yaxes(title = 'Magnitude')
plot(fig)


# 하락장 분석 (11/15)

a = 한달데이터[pd.DatetimeIndex(한달데이터.index).day == 15]  #11월 15일 데이터

fig = px.line(a['Open'], markers=False, title ='갤럭시아머니트리 시가 데이터(11/29)')
fig.update_xaxes(title ='time / min')
fig.update_yaxes(title = 'Open / ₩')
plot(fig)

a_fft = np.fft.fft(a['Open'])
a_fft_M = abs(a_fft)*2/357 # normalization

fig = px.line(a_fft_M[1:], markers=False, title ='갤럭시아머니트리 시가 데이터 FFT(11/29)')
fig.update_xaxes(title ='frequency / Hz')
fig.update_yaxes(title = 'Magnitude')
plot(fig)



#%% 2. STFT 분석

한달데이터시가 = 한달데이터.Open
한달데이터시가.reset_index(inplace=True,drop=True)
선택구간 = 한달데이터시가[5507:5841] # 분명한 상승장 데이터

time_interval = 60
f_s = 1/time_interval

f, t, Zxx = sp.signal.stft(선택구간, fs=f_s, nperseg =20, noverlap=19, nfft=2**8) #STFT
t2m = t/60 # 시간을 분으로

a = np.abs(Zxx)

fig = px.imshow(x=t2m,y=f,img=a, title ='STFT')
fig.update_xaxes(title ='Time / min')
fig.update_yaxes(title = 'frequency / Hz')
plot(fig)



#%% 3. Hilbert Transformation


선택구간 = 선택구간 - 선택구간.mean() # 평균 값 제거

analytic_signal = hilbert(선택구간) # 힐버트 변환
amplitude_envelope = np.abs(analytic_signal) # envelope 성분 추출

fig = px.line(pd.DataFrame({'raw':선택구간, 'ht':amplitude_envelope}), markers=False, title ='한달 데이터')
fig.update_xaxes(title ='Time / min')
fig.update_yaxes(title = 'Openprice / won')
plot(fig)

