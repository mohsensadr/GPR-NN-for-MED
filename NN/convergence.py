import pickle
import numpy as np
import gpflow as gp
from scipy import integrate
from random import randint
from numpy import linalg as LA
from numpy.linalg import inv

import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import tensorflow as tf

import math
import os.path

from numpy import ma

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


def transf_to(xx,m,v,i1,i2):
    xxnew = xx.copy();
    for i in range(i1,i2):
        if len(xx[0]) == 1:
        #    xxnew[i] = (xx[i] - m[i]) / v[i] ** 0.5 + 1.0
        #else:
        #if len(xx[0]) == 1:
            xxnew[:, i] = (xx[:, i] - m) / v ** 0.5 + 1.0
        else:
            xxnew[:,i][0] = (xx[:,i] - m[i] ) / v[i] ** 0.5+1.0
    return xxnew

def transf_back(xx,m,v,i1,i2):
    xxnew = xx.copy();
    for i in range(i1,i2):
        xxnew[:, i][0] = (xx[:,i]-1.0)* v[i] ** 0.5 + m[i]
    return xxnew

size = 8;
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})

mu1 = 0.9#[0.8, 0.9, 0.95]
sig1 = 0.3#[0.3, 0.2, 0.15]
mu2 = -mu1;
sig2 = np.sqrt(2.0 - (sig1 ** 2 + 2 * mu1 ** 2))

xx =  np.linspace(-10.0, 10.0, num=1000)
ff = f(xx,mu1,sig1,mu2,sig2);
fig, ax = plt.subplots();
plt.plot(xx, ff, label=r"$f^{\mathrm{bi}}$",
             linestyle="-", color="black", linewidth=1)



dimXs = [3,4,5,6,8]
dimXs = [3,4,6,8]
#dimXs = [8]
kls = [[] for i in range(len(dimXs))]
kls_ex = [[] for i in range(len(dimXs))]
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

    m_name = "GPR_"+str(dimY)+"/GPR_1000.h5"
    model = gp.saver.Saver().load(m_name)
    suf = ''
    mean_input = np.genfromtxt("GPR_"+str(dimY)+'/GPR_mean_input' + suf + '.dat', dtype=np.float64)
    deviation_input = np.genfromtxt("GPR_"+str(dimY)+'/GPR_variance_input' + suf + '.dat', dtype=np.float64)
    mean_output = np.genfromtxt("GPR_"+str(dimY)+'/GPR_mean_output' + suf + '.dat', dtype=np.float64)
    deviation_output = np.genfromtxt("GPR_"+str(dimY)+'/GPR_variance_output' + suf + '.dat', dtype=np.float64)
    #mean_input = np.array([mean_input])
    #deviation_input = np.array([deviation_input])

    x_predict = np.array([q[2:dimY]])
    x_transfered = transf_to(x_predict, mean_input, deviation_input, 0, dimY-2);
    output, err_output = model.predict_y(x_transfered)
    l = transf_back(output, mean_output, deviation_output, 0, dimY);
    l = l[0]

    q_med_est = Mom(l, dimY, dimX)

    dum = 0.0;
    dum2 = 0.0;
    dum3 =0.0;
    for i in range(dimY):
        dum += (l[i]-l0[i])**2
        dum2 += l0[i]**2
        dum3 += (q_med_ex[i]-q_med_est[i])**2;

    stt += str(dimX) +"   "+str(np.sqrt(dum/dum2))+"   "+str(np.sqrt(dum3))+"\n"

    int0, err = integrate.quad(Z, -1e1, 1e1, args=(l,dimY))
    fMED = Z(xx, l, dimY) / int0;

    int0_ex, err = integrate.quad(Z, -1e1, 1e1, args=(l, dimY))
    fMED_ex = Z(xx, l0, dimY) / int0_ex;

    plt.plot(xx, fMED, label=r'$f^\lambda_{'+str(dimY)+'}$',linewidth=1.0)

    kls[dimXs.index(dimX)] = kl(ff,fMED);
    kls_ex[dimXs.index(dimX)] = kl(ff, fMED_ex);

name_file = "GPR_err.txt"
gg = open(name_file, "w");
gg.write(stt)

plt.xlim(-4.0,4.0)
plt.legend();
ax.set_ylabel(r"$f(x)$")
ax.set_xlabel(r"$x$");
name = "Dist_GPR"+str(dimXs[0])
plt.legend(frameon=False, loc='upper right')
fig.set_size_inches(size*cm, size*cm)
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
name = "KL"
plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
plt.show();


name_file = "GPR_kl.txt"
f = open(name_file, "w");
st=""
for i in range(len(dimXs)):
    st+=str(dimXs[i])+"   "+str(kls[i])+"\n"
f.write(st)

name_file = "ex_kl.txt"
f = open(name_file, "w");
st=""
for i in range(len(dimXs)):
    st+=str(dimXs[i])+"   "+str(kls_ex[i])+"\n"
f.write(st)