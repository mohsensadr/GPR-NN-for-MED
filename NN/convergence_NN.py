
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
# import matplotlib.pyplot as plt
import os
import random
import shutil
import time
from scipy import integrate

import matplotlib.pyplot as plt
def f1(x,mu,sig):
    return 1.0/np.sqrt(2.0*np.pi*sig**2)*np.exp(- (x-mu)**2/(2.0*sig**2));

def f(x,mu1,sig1,mu2,sig2):
    return 0.5*(f1(x,mu1,sig1)+f1(x,mu2,sig2))
def g(x,mu,sig,p):
    return p*f1(x,mu,sig)+(1.0-p)*f1(x,-mu,sig)
def xif(x,mu1,sig1,mu2,sig2,i):
    return x**i*f(x,mu1,sig1,mu2,sig2)
def xig(x,mu,sig,p,i):
    return x**i*g(x,mu,sig,p)

def xif1(x,mu,sig,i):
    return x**i*f1(x,mu,sig)

def f3(x,mu1,sig1,mu2,sig2,mu3,sig3):
    return (1.0/3.0)*(f1(x,mu1,sig1)+f1(x,mu2,sig2)+f1(x,mu3,sig3))
def xif3(x,mu1,sig1,mu2,sig2,mu3,sig3,i):
    return x**i*f3(x,mu1,sig1,mu2,sig2,mu3,sig3)

def Z(v,l,dimY):
    f = 0.0;
    for i in range(1,dimY+1):
        f += l[i-1]*v**i
    return np.exp(-f);

def Hi(v,l,dimY,i):
    return Z(v,l,dimY)*v**i

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def Mom(l, dimY, dimX):
    q = [0 for x in range(dimX)]
    D, err = integrate.quad(Z, -1e1, 1e1, args=(l, dimY));
    for i in range(0, dimX):
        intt, err = integrate.quad(Hi, -1e1, 1e1, args=(l, dimY, i + 1));
        q[i] = intt / D;
    return q

def objective(l, Q, dimY, dimX):
    qq = Mom(l, dimY, dimX);
    sum = 0.0;
    sum += ( Q[0]-qq[0])**2
    for i in range(0,len(qq)):
        if i == 0:
            sum += abs((Q[i] - qq[i]) ) ** 2
        else:
           sum += abs( (Q[i]-qq[i])/qq[i] )**2
    return sum

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(q > 1e-10, p * np.log(p / q), 0))*20.0/len(p);



mu1 = 0.9#[0.8, 0.9, 0.95]
sig1 = 0.3#[0.3, 0.2, 0.15]
mu2 = -mu1;
sig2 = np.sqrt(2.0 - (sig1 ** 2 + 2 * mu1 ** 2))

xx =  np.linspace(-10.0, 10.0, num=1000)
ff = f(xx,mu1,sig1,mu2,sig2);
fig, ax = plt.subplots();
plt.plot(xx, ff, label=r"$f^{\mathrm{ex}}$",
             linestyle="-", color="black", linewidth=1)

dimXs = [3,4,5,6,8]
dimXs = [3,4,6]
dimXs = [8]
kls = [[] for i in range(len(dimXs))]
#dimXs = [12]
stt = ""
for dimX in dimXs:
    q = []
    dimY = dimX
    for i in range(0, dimX):
        I, d2 = integrate.quad(xif, -1e1, 1e1, args=(mu1, sig1, mu2, sig2, i + 1));
        # I, d2 = integrate.quad(xig, -1e1, 1e1, args=(mu, sig, p, i + 1));
        q.append(I);
    q = np.array(q)
    if dimX == 3:
       l0 = [-0.1236756900314129,  0.516679580252206, 0.04122636998331299]
    elif dimX == 4:
       l0 = [-0.7043640854047318, -1.213225498156559, 0.432670353803256, 0.5496731259305444]
    elif dimX == 5:
       l0 = [-0.5868253785687041, -1.404746238432348, 0.2542218631713953, 0.6364634545841446, 0.059708178927589256]
    elif dimX == 6:
       l0 = [-0.04743950102790144,-1.7963760730592575,-0.5575983379196991,0.7722133489203675,0.32662138992668466,0.03369916619152445]
    elif dimX == 8:
        l0 = [-0.21478409567528478, -3.0224417983240226, -0.24331151837042694,  2.0160747438695665, 0.26935442896527784, -0.335545383359783, -0.029866218468154883,  0.018967782958135355]
    elif dimX == 10:
        l0 = [0.2358568866960999, -3.7058601617217675, -1.5867611462753994, 2.8795987352013457,  1.47554313968298, -0.6247812361167333,
               -0.4413939946171417, 0.018899325852642597,  0.04591438724401074, 0.006652004065273113]
    elif dimX == 12:
        l0 = [-8.36219439e-01, -3.98330026e+00,  1.24064376e+00,  3.44102595e+00,
 -6.74425845e-01, -9.71134961e-01,  1.95909494e-01,  5.81724219e-02,
 -5.88607262e-02,  1.10530072e-02,  1.43841813e-02,  2.12195425e-03]

    q_med_ex = Mom(l0, dimY, dimX)

    print(q_med_ex)
    mad = 'functions_' + str(dimX)
    model = keras.models.load_model(mad + '/my_model.h5')

    ## Load the feature scaling related arrays

    mean_input = np.genfromtxt(mad + '/mean_input.dat', dtype=np.float64)
    deviation_input = np.genfromtxt(mad + '/deviation_input.dat', dtype=np.float64)
    mean_output = np.genfromtxt(mad + '/mean_output.dat', dtype=np.float64)
    deviation_output = np.genfromtxt(mad + '/deviation_output.dat', dtype=np.float64)

    x_predict = np.array([q[2:dimY]])

    for j in range(len(x_predict[0, :])):
        if dimX != 3:
            x_predict[:, j][0] = (x_predict[:, j][0] - mean_input[j]) / deviation_input[j]
        else:
            x_predict[:, j][0] = (x_predict[:, j][0] - mean_input) / deviation_input

    output = model.predict(x_predict)
    for j in range(len(output[0, :])):
        output[:, j] = output[:, j] * deviation_output[j] + mean_output[j]
    l = output[0]

    q_med_est = Mom(l, dimY, dimX)

    dum = 0.0;
    dum2 = 0.0;
    dum3 = 0.0;
    for i in range(dimY):
        dum += (l[i] - l0[i]) ** 2
        dum2 += l0[i] ** 2
        dum3 += (q_med_ex[i] - q_med_est[i]) ** 2;

    stt += str(dimX) + "   " + str(np.sqrt(dum / dum2)) + "   " + str(np.sqrt(dum3)) + "\n"


    print("error = "+str(l-l0))
    int0, err = integrate.quad(Z, -1e1, 1e1, args=(l,dimY))
    fMED = Z(xx, l, dimY) / int0;
    plt.plot(xx, fMED, label='N='+str(dimY))

    kls[dimXs.index(dimX)] = kl(ff,fMED);

'''
name_file = "NN_err.txt"
gg = open(name_file, "w");
gg.write(stt)
'''
plt.xlim(-3.0,3.0)
plt.legend();
name = "Dist_NN"
plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
plt.show();



fig, ax = plt.subplots();
plt.plot(dimXs,kls, marker="o")
plt.legend(frameon=False,loc="lower left")
ax.set_ylabel(r"$D_{KL}(f^{\mathrm{bi}}||f^\lambda_N)$")
ax.set_yscale("log")
plt.xticks(dimXs)
plt.ylim(1e-3,1e0)
#fig.set_size_inches(size * cm, size * cm)
plt.legend(frameon=False, loc='lower left')
name = "KL_NN"
plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
plt.show();

'''
name_file = "NN_kl.txt"
f = open(name_file, "w");
st=""
for i in range(len(dimXs)):
    st+=str(dimXs[i])+"   "+str(kls[i])+"\n"
f.write(st)
'''