# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:38:29 2023

@author: ADMIN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
from plotly.offline import plot
import plotly.offline as pyo
import plotly.graph_objs as go

N = 256
dt = 0.1 #단위: s
time = np.arange(1,N+1)*dt
T = N*dt #주기
signal = np.ones(N)
signal[N//3:] = -np.ones(N-N//3)

plt.figure(figsize=(8,2))
plt.scatter(time,signal)
plt.grid(True)


#%%
signal_fft = np.fft.fft(signal)
signal_fft_M = np.sqrt((signal_fft.real)**2+(signal_fft.imag)**2)
# magnitude = np.abs(signal_fft)

plt.figure(figsize=(32,8))
plt.bar(np.arange(1,129),signal_fft_M[1:129])
plt.grid(True)


#%%

def m_k (k):
    return (2/(k*np.pi))*np.sqrt(2-2*np.cos(k*(2*np.pi/3)))


for i in range(1,5):
    print(f"M_{i} is ... "  + str(m_k(i)))


#%% sin 함수의 연속 푸리에 계수, DFT 비교

N = 4
dt = 1 #단위: s
time = np.arange(0,N)*dt
T = N*dt #주기
signal = np.cos(2*np.pi/T*time)

plt.figure(figsize=(8,2))
plt.plot(time,signal)

signal_fft = np.fft.fft(signal)
signal_fft_M = np.sqrt((signal_fft.real)**2+(signal_fft.imag)**2)
signal_ifft = np.fft.ifft(signal_fft)
signal_ifft_M = np.abs(signal_ifft)

plt.plot(signal_ifft_M)

#%%
fig = px.line(x = time, y=a, markers=True, title ='사각파')
fig.update_xaxes(title ='Time / s').update_yaxes(title = 'Value')
#pyo.iplot(fig)
plot(fig)

def a_n(T, n):
    return (2/T)*2*np.sin(np.pi*(2/3)*n)

def b_n(T,n):
    return (2/T)*2*(1-np.cos(np.pi*(2/3)*n))

def M_n(n):
    a_n =  (2/25.6)*2*np.sin(np.pi*(2/3)*n)
    b_n =  (2/25.6)*2*(1-np.cos(np.pi*(2/3)*n))
    return np.sqrt(a_n**2+b_n**2)

number  = np.arange(1,256)
M_n = M_n(number)

fig = px.line(x = number, y =M_n, markers=True, title ='$M_n$')
fig.update_xaxes(title ='number').update_yaxes(title = 'Value')
plot(fig)

X = np.fft.fft(a)
power_spectrum = np.abs(X)

frequency = number*(1/T)
fig = px.line(x=frequency, y =power_spectrum, title ='$$M_n(DFT)$$')
fig.update_xaxes(title ='freqeuncy / Hz').update_yaxes(title = 'Value')
plot(fig)


df = pd.DataFrame({'analytical one':M_n,'DFT':power_spectrum[1:]})
df.index =number
fig = px.line(df, title ='$$M_n(DFT)$$', markers=True)
fig.update_xaxes(title ='number').update_yaxes(title = 'Value')
plot(fig)
