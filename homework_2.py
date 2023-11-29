# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:38:29 2023

@author: ADMIN
"""

import pandas as pd
import numpy as np

import plotly.express as px
from plotly.offline import plot
import plotly.offline as pyo
import plotly.graph_objs as go

N = 1024
dt = 0.001 #단위: s
time = np.arange(1,N+1)*dt

signal = np.sin(2*np.pi*10*time)

fig = px.line(x = time, y=signal, markers=True, title ='사인파')
fig.update_xaxes(title ='Time / s').update_yaxes(title = 'Value')
plot(fig) #pyo.iplot(fig)


N_256 = np.fft.fft(signal,N)
power_spectrum = np.abs(N_256)**2
T = time[-1]
frequency = np.arange(0,N)*(1/T)

fig = px.line(x=frequency, y =power_spectrum, title ='sin-DFT-256')
fig.update_xaxes(title ='freqeuncy / Hz').update_yaxes(title = 'Value')
plot(fig)

mean_square_value = (np.abs(N_256) ** 2).sum()/N
print(f'N={N}에 대한 평균 제곱 값: {mean_square_value}')

#%%
import numpy as np
import matplotlib.pyplot as plt

# 신호 정의
t = np.arange(0, 1, 0.001)
x = np.sin(20 * np.pi * t)

# N 값 정의
N_values = [512, 1024, 2024]

# 각 N에 대한 파워 스펙트럼 플로팅
for N in N_values:
    # 푸리에 변환 수행
    freq = np.fft.fftfreq(N, 0.001)
    x_fft = np.fft.fft(x[:N])
    
    # 파워 스펙트럼 계산
    power_spectrum = np.abs(x_fft) ** 2
    
    # 파워 스펙트럼 플로팅
    plt.plot(freq, power_spectrum[:N], label=f'N={N}')

# 플롯 매개변수 설정
plt.xlabel('주파수 (Hz)')
plt.ylabel('파워')
plt.title('다른 N에 대한 파워 스펙트럼')
plt.legend()
plt.show()
