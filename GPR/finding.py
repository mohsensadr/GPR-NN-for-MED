import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
from scipy import integrate


def f1(x,mu,sig):
    return 1.0/np.sqrt(2.0*np.pi*sig**2)*np.exp(- (x-mu)**2/(2.0*sig**2));
def f(x,mu1,sig1,mu2,sig2):
    return 0.5*(f1(x,mu1,sig1)+f1(x,mu2,sig2))
def xif(x,mu1,sig1,mu2,sig2,i):
    y= x**i*f(x,mu1,sig1,mu2,sig2)
    return y

mu = [0.8, 0.9, 0.95]
sig = [0.3, 0.2, 0.15]

case = 0;
mu1 = mu[case]  # + ((mu_end-mu_beg)/(n-1))*j
mu2 = -mu1;
sig1 = sig[case];
sig2 = np.sqrt(2.0 - (sig1 ** 2 + 2 * mu1 ** 2))

dimY = 8
val = 1
p_ex = np.zeros(dimY);
for i in range(0, dimY):
    p_ex[i], d2 = integrate.quad(xif, -1e1, 1e1, args=(mu1, sig1, mu2, sig2, i + 1));

address = "expansion/"+str(dimY)+"_"+str(val)+".txt"
x = np.loadtxt(address,skiprows=1, unpack=True);
N = len(x[0])-1
print(str(N)+" data points")
Q = []
La = []
i0=0; i1=dimY; i2=len(x[:,0])

accept = 1
for i in range(0,N):
    q = x[:, i][i0:i1]
    l = x[:, i][i1:i2]
    if( np.linalg.norm(q[2:dimY]-p_ex[2:dimY])/np.linalg.norm(p_ex[2:dimY]) < 1e-2 ):
        print(np.linalg.norm(q-p_ex))
