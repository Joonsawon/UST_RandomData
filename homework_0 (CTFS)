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

N = 256
dt = 0.1 #단위: s
time = np.arange(1,N+1)*0.1 
T = N*dt
a = np.ones(N)
a[N//3:] = -np.ones(N-N//3)

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
