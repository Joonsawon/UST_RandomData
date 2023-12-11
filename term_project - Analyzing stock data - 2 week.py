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

#%% Mean value, Autocorrelation 계산
# 한달 데이터를 앙상블로 함.

# 1. 데이터 불러오기
한달데이터 = pd.read_csv('C:/GitHub/UST_RandomData/갤럭시아머니트리.csv',index_col=0) #인덱스로 사용할 컬럼 설정

# 첫번째 데이터 불러오기
# 9시 0분 데이터...
첫데이터들 = 한달데이터[(pd.DatetimeIndex(한달데이터.index).hour == 9) & (pd.DatetimeIndex(한달데이터.index).minute == 0)] 
# 그런데 이런식으로 하면 나머지 데이터들을 불러올 때 너무 힘들 것 같다. 
# 하루 데이터의 수를 카운트 해서 첫번째+데이터 수 이런식으로 불러오는 것이 반복문 처리하기에 좋을 듯 하다.
# 하루 데이터 수는

첫날데이터 = 한달데이터[pd.DatetimeIndex(한달데이터.index).day == 4]  #12월 4일 데이터
둘째데이터 = 한달데이터[pd.DatetimeIndex(한달데이터.index).day == 7]  #11월 7일 데이터
# 한달데이터.index → '2023-11-07 09:00:00+09:00' 와 같은 정보를 출력
# pd.DatetimeIndex(한달데이터.index).day → index 중 day에 해당하는 int 값을 출력
# pd.DatetimeIndex(한달데이터.index).day == 7 → index 중 day 값이 7인 low만 True, 나머지는 False
# 한달데이터[pd.DatetimeIndex(한달데이터.index).day == 7] → index.day 값이 7인 정보만 출력
# 7,8일 357개. 4일 358개. .... 그때 그 때 다르냐...
# 어디가 다른걸까
# 근데 60*6 = 360개가 되어야 하는데 한두개 씩 데이터가 빠진 것 같다..ㅎ

# 첫 데이터들을 살펴보니 19개 밖에 없다. 28개여야 하는데.. 
# 9시 1분 데이터를 보자
둘데이터들 = 한달데이터[(pd.DatetimeIndex(한달데이터.index).hour == 9) & (pd.DatetimeIndex(한달데이터.index).minute == 1)] 

#가 아니고 평일만 치면 20일 인데 16일에 수능 때문에 10시부터 장 시작.. 16일 데이터는 제거하고 가자. 
# 그러면 19일 데이터들의 시작 데이터들의 평균은
첫데이터들.Open.mean() # 6997.368421052632 이 나온다. 


#%% t에 따른 Mean Value 그래프를 그려보자

# 일단 날짜 인덱스를 저장하자
날짜들 = pd.DatetimeIndex(첫데이터들.index).day

#날짜별로 정제한 데이터 중 1번째 데이터만 수집. 
#시간으로 하면 편하지만, 날짜별로 없는 시간대가 있기 때문에 이 방식으로 진행.
# 우선, 첫번째 데이터들의 평균
저장소 = [] # 날짜별 첫번째 데이터를 담을 리스트 변수 생성
for i in 날짜들: 
    하루데이터 = 한달데이터[pd.DatetimeIndex(한달데이터.index).day == i] # i번째 날의 데이터 받기
    저장소.append(하루데이터.Open[0]) # Open 데이터 중 첫번째 데이터만 담기

저장소 = pd.DataFrame(저장소)
저장소.mean()


#%% 이제 시간순으로 300개 데이터의 Mean Value를 계산하고 그래프 그려보자


평균값 = [] # 시간별 앙상블 평균을 담을 리스트 변수

for j in range(300):
    저장소 = [] # 날짜별 첫번째 데이터를 담을 리스트 변수 초기화
    for i in 날짜들: 
        하루데이터 = 한달데이터[pd.DatetimeIndex(한달데이터.index).day == i] # i번째 날의 데이터 받기
        저장소.append(하루데이터.Open[j]) # Open 데이터 중 첫번째 데이터만 담기 (좀 비효율적이긴 하네)
    평균값.append(pd.DataFrame(저장소).mean()[0])
# 오래걸림 ㅋㅋㅋ 그냥 넘파이로 할걸


#%%# 데이터프레임으로 받아서 인덱스별 평균을 내보자

저장소 = pd.DataFrame([]) # 날짜별 첫번째 데이터를 담을 리스트 변수 초기화
for i in 날짜들: 
    하루데이터 = 한달데이터[pd.DatetimeIndex(한달데이터.index).day == i] # i번째 날의 데이터 받기
    하루데이터.reset_index(inplace=True,drop=True)
    저장소[f'{i}']=하루데이터.Open[0:300] # Open 데이터 중 첫번째 데이터만 담기 (좀 비효율적이긴 하네)
    
# 첫째날을 제외하고 NaN이 뜨는 문제 발생. 아마도 index가 안맞아서 그런듯. index제거를 추가하자.
# 성공!

#%% 이제 행별 평균을 구해보자.

mean_value = 저장소.mean(axis='columns') # 300개 데이터가 나오는 것을 확인할 수 있다. 

#그래프를 그려볼까?
fig = px.line(mean_value, markers=True, title ='장 시간에 따른 주가의 앙상블 평균')
fig.update_xaxes(title ='index')
fig.update_yaxes(title = 'Mean value / ₩')
plot(fig)



#%% 주가 데이터 전체 날짜별 플롯


fig = px.line(저장소, markers=False, title ='날짜별 주가 추이')
fig.update_xaxes(title ='index')
fig.update_yaxes(title = 'Open price / ₩')
plot(fig)


#%% 날짜별 변동률(시작가 기준)

변동률 = pd.DataFrame([]) # 날짜별 첫번째 데이터를 담을 리스트 변수 초기화
for i in 날짜들: 
    하루데이터 = 한달데이터[pd.DatetimeIndex(한달데이터.index).day == i] # i번째 날의 데이터 받기
    하루데이터.reset_index(inplace=True,drop=True)
    변동률[f'{i}']=하루데이터.Open[0:300]/하루데이터.Open[0]*100 # 매일의 시가로 나눔
    

fig = px.line(변동률, markers=False, title ='날짜별 주가 추이')
fig.update_xaxes(title ='index')
fig.update_yaxes(title = 'Open price / ₩')
plot(fig)


# 변동률의 앙상블 평균은?

변동률_mean_value = 변동률.mean(axis='columns') # 300개 데이터가 나오는 것을 확인할 수 있다. 

#그래프를 그려볼까?
fig = px.line(변동률_mean_value, markers=True, title ='장 시간에 따른 주가 변동률의 앙상블 평균')
fig.update_xaxes(title ='index')
fig.update_yaxes(title = 'Change rate / %')
plot(fig)





#%% STFT

# 상승장이 확실한 29일 데이터 불러오기

하루데이터_29 = 한달데이터[pd.DatetimeIndex(한달데이터.index).day == 29].Open
하루데이터_29.reset_index(inplace=True,drop=True) # 날짜 인덱스 제거
하루데이터_29= 하루데이터_29[0:256]

하루데이터_29_stft = sp.signal.stft(하루데이터_29)

import matplotlib.pyplot as plt

f, t, Zxx = sp.signal.stft(하루데이터_29,fs=0.1666666,nperseg =5, nfft =100)
a= np.abs(Zxx)
fig = px.imshow(img=np.abs(Zxx), title ='STFT')
fig.update_xaxes(title ='Time')
fig.update_yaxes(title = 'frequency / Hz')
plot(fig)



#%% 하루는 데이터가 300 여개라 정보가 부족하다.. 한달 전체의 데이터에 대해 분석해보자

한달데이터시가 = 한달데이터.Open
한달데이터시가.reset_index(inplace=True,drop=True)

fig = px.line(한달데이터시가, markers=False, title ='시간별 주가 추이')
fig.update_xaxes(title ='Time / min')
fig.update_yaxes(title = 'Open price / ₩')
plot(fig)


f, t, Zxx = sp.signal.stft(한달데이터시가,fs=0.1666666,nperseg =30)
fig = px.imshow(img=np.abs(Zxx), title ='STFT')
fig.update_xaxes(title ='Time')
fig.update_yaxes(title = 'frequency / Hz')
plot(fig)

import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(z=np.abs(Zxx))])
plot(fig)

fig = px.scatter_3d(np.abs(Zxx), title ='STFT')
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