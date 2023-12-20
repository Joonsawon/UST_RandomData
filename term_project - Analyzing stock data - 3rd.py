# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 07:20:26 2023

@author: ADMIN
"""
#import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.offline import plot
import scipy as sp
from scipy.signal import hilbert, chirp

#%% Mean value, Autocorrelation 계산
# 한달 데이터를 앙상블로 함.

# 1. 데이터 불러오기
한달데이터 = pd.read_csv('C:/GitHub/UST_RandomData/갤럭시아머니트리.csv',index_col=0) #인덱스로 사용할 컬럼 설정


첫날데이터 = 한달데이터[pd.DatetimeIndex(한달데이터.index).day == 20]  #12월 4일 데이터

# 한달데이터.index → '2023-11-07 09:00:00+09:00' 와 같은 정보를 출력
# pd.DatetimeIndex(한달데이터.index).day → index 중 day에 해당하는 int 값을 출력
# pd.DatetimeIndex(한달데이터.index).day == 7 → index 중 day 값이 7인 low만 True, 나머지는 False
# 한달데이터[pd.DatetimeIndex(한달데이터.index).day == 7] → index.day 값이 7인 정보만 출력



# 지난번 분석했던 급등 날의 힐버트 변환을 수행. 일단 마무리하고 여유가 되면 
# csv로 받아서 gpt에 입력해보자.

한달데이터시가 = 한달데이터.Open
한달데이터시가.reset_index(inplace=True,drop=True)


time_interval = 60
f_s = 1/time_interval
선택구간 = 한달데이터시가[5507:5841]

선택구간 = 첫날데이터.Open

f, t, Zxx = sp.signal.stft(선택구간, fs=f_s, nperseg =20, noverlap=19, nfft=2**8 )
t2m = t/60

a = np.abs(Zxx)
#a = a[2:,1:-2]
f2mf = f*10000

fig = px.imshow(x=t2m,y=f2mf,img=a, title ='STFT')
fig.update_xaxes(title ='Time / min')
fig.update_yaxes(title = 'frequency / Hz')
plot(fig)



fig = px.line(선택구간, markers=False, title ='한달 데이터')
fig.update_xaxes(title ='Time / min')
fig.update_yaxes(title = 'Openprice / won')
plot(fig)

선택구간 = 선택구간 - 선택구간.mean()

analytic_signal = hilbert(선택구간)
amplitude_envelope = np.abs(analytic_signal)


fig = px.line(pd.DataFrame({'raw':선택구간, 'ht':amplitude_envelope}), markers=False, title ='한달 데이터')
fig.update_xaxes(title ='Time / min')
fig.update_yaxes(title = 'Openprice / won')
plot(fig)



#%% 무슨 의미인지는 잘 모르겠다. 우선 x,y 스케일을 잘 맞춰보자. 


fig = px.line(한달데이터시가, markers=False, title ='한달 데이터')
fig.update_xaxes(title ='Time / min')
fig.update_yaxes(title = 'Openprice / won')
plot(fig)

#. 1. x,y 스케일을 맞춰주고,
# 2. 급등하는 구간의 데이터를 분석해보자. 

#%%

time_interval = 60
f_s = 1/time_interval
선택구간 = 한달데이터시가[5507:5841]

f, t, Zxx = sp.signal.stft(한달데이터시가[5507:5841], fs=f_s, nperseg =20, noverlap=19, nfft=2**8 )
t2m = t/60

a = np.abs(Zxx)
#a = a[2:,1:-2]
f2mf = f*10000

fig = px.imshow(x=t2m,y=f2mf,img=a, title ='STFT')
fig.update_xaxes(title ='Time / min')
fig.update_yaxes(title = 'frequency / Hz')
plot(fig)


# 노멀라이즈 해서 비교해보자

fig = px.imshow(x=t2m,y=f2uf,img=a/a.max(), title ='STFT')
fig.update_xaxes(title ='Time / min')
fig.update_yaxes(title = 'frequency / Hz')
plot(fig)

sp.signal. check_COLA ( window = 'hann' , nperseg = 20 , noverlap=19 , tol = 1e-10 )


# 5507-5840
# 6744-6767
# 





















#%%









#%%

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

a = concat[pd.DatetimeIndex(concat.index).day == 20]  #11월 7일 데이터

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