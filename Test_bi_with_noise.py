import numpy as np
from scipy import integrate
import os
import time
from scipy.optimize import minimize
from scipy import integrate
from numpy import linalg as LA
#from scipy.integrate import trapz

def f1(x,mu,sig):
    return 1.0/np.sqrt(2.0*np.pi*sig**2)*np.exp(- (x-mu)**2/(2.0*sig**2));
def f(x,mu1,sig1,mu2,sig2):
    return 0.5*(f1(x,mu1,sig1)+f1(x,mu2,sig2))
def xif(x,mu1,sig1,mu2,sig2,i):
    y= x**i*f(x,mu1,sig1,mu2,sig2)
    return y

def Z(v,l,dimY):
    f = 0.0;
    for i in range(1,dimY+1):
        f += l[i-1]*v**i
    return np.exp(-f);
def Hi(v,l,dimY,i):
    return Z(v,l,dimY)*v**i
def Mom(l, dimY, dimX):
    q = [0 for x in range(dimX)]
    D, err = integrate.quad(Z, -1e1, 1e1, args=(l, dimY));
    for i in range(0, dimX):
        intt, err = integrate.quad(Hi, -1e1, 1e1, args=(l, dimY, i + 1));
        q[i] = intt / D;
    return q;





### read GPR models
import gpflow as gp
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import tensorflow as tf
def transf_to(xx,m,v,i1,i2):
    xxnew = xx.copy();
    for i in range(i1,i2):
        xxnew[:,i] = (xx[:,i] - m[i] ) / v[i] ** 0.5+1.0
    return xxnew

def transf_back(xx,m,v,i1,i2):
    xxnew = xx.copy();
    for i in range(i1,i2):
        xxnew[:, i] = (xx[:,i]-1.0)* v[i] ** 0.5 + m[i]
    return xxnew

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

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
size = 6;
lw = 1
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})
colors = ['black', 'r', 'b', 'g', 'm', 'c']
markers = ["<","s","o"]

fig, ax = plt.subplots();
#stat = "var"
#data_address = ["data/4l5e-2_"+stat+"4th.txt", "data/6l1e-4_"+stat+"6th.txt", "data/8l1e-7_"+stat+"8th.txt"]
#model_names = ["models/4_"+stat+"_5e-2_1000.txt", "models/6_"+stat+"_1e-4_1000.txt", "models/8_"+stat+"_1e-7_1000.txt"]
data_address = ["data/4l.txt", "data/6l.txt", "data/8l.txt"]
model_names = ["models/4l_1000.txt", "models/6l_1000.txt", "models/8l_1000.txt"]



#plt.plot(xx, ff, label=r"$f^{\mathrm{bi}}$",
#             linestyle="-", color="black", linewidth=1)


models = []
for j in range(len(model_names)):
    m_name = model_names[j]
    models.append(gp.saver.Saver().load(m_name))

mu = [0.8, 0.9, 0.95]
sig = [0.3, 0.2, 0.15]
var_noise = 1e-1;

case = 2;
mu1 = mu[case]  # + ((mu_end-mu_beg)/(n-1))*j
mu2 = -mu1;
sig1 = sig[case];
sig2 = np.sqrt(2.0 - (sig1 ** 2 + 2 * mu1 ** 2))

xx = np.arange(-10.0, 10.0, 0.001)
ff = f(xx,mu1,sig1,mu2,sig2);
## add noise
rrs = 1;
lla = ["(a)","(b)","(c)"]
La = []; [La.append([]) for _ in range(len(model_names))];
DKL_val = np.zeros(len(model_names));# [DKL_val.append([]) for _ in range(len(model_names))]
varrr = np.zeros(len(model_names));# [varrr.append([]) for _ in range(len(model_names))]
pp = np.zeros(len(model_names))#[]; [pp.append([]) for _ in range(len(model_names))]

for rr in range(rrs):
    ffe = ff*(1.0 + np.random.normal(0.0,var_noise,len(xx)));

    plt.plot(xx, ffe, label=r"$f^{\mathrm{bi}}_\epsilon$",
                 linestyle="-", color="grey", linewidth=0.1)
    dimY = 8;
    pe = np.zeros(dimY); p = np.zeros(dimY);
    for i in range(dimY):
        p[i] = integrate.trapz(xx ** (i + 1) * ff, xx);
        pe[i] = integrate.trapz(xx ** (i + 1) * ffe, xx);



    for j in range(len(model_names)):
        address = data_address[j]
        x0 = np.loadtxt(address, skiprows=1, unpack=True);
        dim = len(x0[:, 0]);  ## 26 here

        vvar = []
        mm = []
        for i in range(dim):
            mm.append(np.mean(x0[i, :]));
            vvar.append(np.var(x0[i, :]));
        model = models[j]#gp.saver.Saver().load(m_name)

        N1 = len(model.X.value)
        dimX = len(model.X.value[0])
        dimY = len(model.Y.value[0]);

        mmX = mm[0:dimX + 2]
        varX = vvar[0:dimX + 2]
        mmY = mm[20:]
        varY = vvar[20:]

        q = [pe];
        q = np.array(q)
        # scale input
        q_sc = transf_to(q, mmX, varX, 2, dimX + 2);
        # predict
        ystar2, varstar2 = model.predict_y(q_sc[:, 2:dimX + 2])
        print("mu1=" + str(mu1) + " var = " + str(varstar2[0, 0]))
        # transfer back
        q_tf = transf_back(q_sc, mmX, varX, 2, dimX + 2);
        la_tf = transf_back(ystar2, mmY, varY, 0, dimY);
        la_tf = la_tf[0]
        La[j].append(la_tf);
        I, d = integrate.quad(Z, -1e1, 1e1, args=(La[j][0],dimY));

        fl = Z(xx, la_tf, dimY) / I;
        plt.plot(xx, fl, label=r"$f^{\lambda}_"+str(dimY)+"$", marker=markers[j], markevery=400+j*10,
             markersize=3, color=colors[j + 1], linewidth=lw)

        p = Mom(la_tf, dimY, dimY);
        DKL_val[j] += kl(ffe, fl);
        varrr[j] += varstar2[0, 0]
        pp[j] += np.linalg.norm(p-pe[0:dimY])/np.linalg.norm(pe[0:dimY])

plt.text(0.1, 0.9, lla[case], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

'''
fp = open("kl_mu1_"+str(mu1), "w")
st = ""
for j in range(2,5):
    st+=str(j*2)+"  "+str(DKL_val[j-2]/rrs)+"   "+str(varrr[j-2]/rrs)+"   "+str(pp[j-2]/rrs)+"\n"
fp.write(st);
fp.close()
'''
plt.legend(frameon=False, bbox_to_anchor=(1.02, 1.02),  ncol=1)

'''
for dimY in [4,6,8]:
    x = np.genfromtxt('direct/dimY_'+str(dimY)+'_mu1_'+str(mu1), dtype=np.float64);
    lex = x[dimY:2*dimY]
    I, d = integrate.quad(Z, -1e1, 1e1, args=(lex, len(lex)));
    fex = Z(xx, lex, len(lex)) / I
    plt.plot(xx, fex, label=r"$f^{\mathrm{ex.}}_"+str(dimY)+"$", linestyle="--")
'''
plt.xlim(-3.0, 3.0)
#plt.ylim(-0.1, 2.0)

ax.set_ylabel(r"$f(x)$")
ax.set_xlabel(r"$x$")
fig.set_size_inches(size * cm, size * cm)
plt.savefig("TestCase1_with_noise/"+lla[case]+".pdf", format='pdf', bbox_inches="tight", dpi=300);
ax.legend();
plt.show()