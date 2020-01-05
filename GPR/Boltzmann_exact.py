from mpmath import *
mp.dps = 25; mp.pretty = True

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from scipy import integrate
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
size = 5;
lw = 1
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})
colors = ['black', 'r', 'b', 'g', 'm', 'c', 'c', 'c', 'c']
markers = ["<","s","o",">","o", "o","o","o","o" ]

'''
def F(v):
    x = v ** 2 / 2.0;
    N = len(x)
    y = np.zeros(len(x))
    for i in range(N):
        dummy = -1.0+(1+x[i])*np.exp(x[i])*e1(x[i])
        y[i] = (2.0*np.pi**5.0)**(-0.5)*dummy
    return y

def f(v,t):
    return np.exp(-t)*F(v*np.exp(-t/3.0));
'''
'''
def dl(mu):
    g = 1.0;
    return g*(1.0-mu**2.0)*np.pi/2.0
def f(v,t):
    l, err = integrate.quad(dl, -1.0, 1.0);
    theta = 2.0/5.0;
    tau = 1-theta*np.exp(-l*t)
    dummy = 1.0+(1.0-tau)/tau*(v**2.0/2.0/tau-1.5)
    return (2.0*np.pi*tau)**(-1.5)*np.exp(-v**2.0/(2.0*tau))*dummy
'''
def dphi1(X):
    return 1.0*(np.sin(X))**3
def f(v,tau):
    phi1, err = integrate.quad(dphi1, 0.0, np.pi);
    phi1 = 3.0/4.0*phi1
    phi1 = 1.0
    alpha = 1.0-np.exp(-phi1*tau/6.0)
    dummy = (5.0*alpha-3.0)/2.0/alpha + (1.0-alpha)/(2.0*alpha**2)*v**2
    return (2.0*np.pi*alpha)**(-1.5)*np.exp(-v**2.0/(2.0*alpha))*dummy

def Hf(v,tau,i):
    return f(v,tau)*v**i;
def Mn(n,tau):
    m = np.zeros(n)
    p = np.zeros(2*n)
    phi1 = 1.0
    alpha = 1.0 - np.exp(-phi1 * tau / 6.0)
    for nn in range(n):
        m[nn] = alpha**(nn-1)*(nn-(nn-1.0)*alpha)
        p[2*nn+1] = math.factorial(2*nn+1)/(2**nn*math.factorial(nn))*m[nn]
    return p
def Mn_num(n,tau):
    m = np.zeros(n);
    I, err = integrate.quad(Hf, -10.0, 10.0, args=(tau, 0));
    for nn in range(n):
        m[nn], err = integrate.quad(Hf, -10.0, 10.0, args=(tau,nn+1))
        m[nn] = m[nn]/I
    return m

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

data_address = ["data/4l.txt", "data/6l.txt", "data/8l.txt"]
model_names = ["models/4l_1000.txt", "models/6ln_1000.txt", "models/8ln_1000.txt"]
models = []
for j in range(len(model_names)):
    m_name = model_names[j]
    models.append(gp.saver.Saver().load(m_name))
xx = np.arange(-10.0, 10.0, 0.001)
tt = [5.8, 6.5, 7.5, 8.5, 10, 15,  20.0]
tt = [5.8, 6.5, 7.5, 8.5, 9.5, 10.5]
tt = [5.8, 6.5, 7.5, 8.5, 9.0]
#k = 0
gp = open("Boltz_ex_mom.txt", "w")
for k in range(len(tt)):
    t = tt[k]
    fig, ax = plt.subplots();


    m_num = Mn_num(2, t)
    st = m_num[1] ** (0.5)
    f0 = f(xx*st, t);
    I, d = integrate.quad(f, -1e1, 1e1, args=(t));
    I = integrate.trapz(f0,xx)
    f0 = np.array(f0) / I;
    plt.plot(xx, f0, label=r"$\hat{f}^{\mathrm{Bolt}}$",
         linestyle="-", color="black", linewidth=1.0)

    DKL_val = []; [DKL_val.append([]) for _ in range(len(tt))]
    p_ex = []; [p_ex.append([]) for _ in range(len(model_names))]
    p_pr = []; [p_pr.append([]) for _ in range(len(model_names))]
    for j in range(len(model_names)):
        model_num = j

        ######   read lambdas from GPR
        address = data_address[model_num]
        x0 = np.loadtxt(address, skiprows=1, unpack=True);
        dim = len(x0[:, 0]);  ## 26 here

        vvar = []
        mm = []
        for i in range(dim):
            mm.append(np.mean(x0[i, :]));
            vvar.append(np.var(x0[i, :]));
        model = models[model_num]  # gp.saver.Saver().load(m_name)

        N1 = len(model.X.value)
        dimX = len(model.X.value[0])
        dimY = len(model.Y.value[0]);

        mmX = mm[0:dimX + 2]
        varX = vvar[0:dimX + 2]
        mmY = mm[20:]
        varY = vvar[20:]
        ##################################3
        m_num = Mn_num(dimY, t)
        ## map to standardized moments
        for i in range(len(m_num)):
            m_num[i] = m_num[i] / (st ** (i + 1.0))
        ################################
        mn = Mn(int(dimY/2),t)
        for i in range(0,len(mn)):
            mn[i] = mn[i] / (st ** ((i - 1.0)))
        print(m_num)
        q = [m_num];
        q = np.array(q)
        # scale input
        q_sc = transf_to(q, mmX, varX, 2, dimX + 2);
        # predict
        ystar2, varstar2 = model.predict_y(q_sc[:, 2:dimX + 2])
        print("t=" + str(t) + " var = " + str(varstar2[0, 0]))
        # transfer back
        q_tf = transf_back(q_sc, mmX, varX, 2, dimX + 2);
        la_tf = transf_back(ystar2, mmY, varY, 0, dimY);
        la_tf = la_tf[0]
        #La[j].append(la_tf);
        I, d = integrate.quad(Z, -1e1, 1e1, args=(la_tf, dimY));

        fl = Z(xx, la_tf, dimY) / I;
        plt.plot(xx, fl, label=r"$f^{\lambda}_" + str(dimY) + "$", marker=markers[j], markevery=500+10*j,
             markersize=3, color=colors[j + 1], linewidth=lw)

        q_pred = Mom(la_tf, dimY, dimY)
        DKL = kl(f0,fl)
        '''
        fp = open("Boltz_dimY_" + str(4 + 2 * (model_num)), "a")
        stt = str(t) + "  " + str(DKL) + "   "
        for kk in range(len(q_pred)):
              stt += str(q_pred[kk])+"  "
        stt+="\n"
        fp.write(stt);
        fp.close()
        '''
        ## map standard to original moments
        #for i in range(len(m_num)):
        #    m_num[i] = m_num[i]*(st**(i+1.0))
    ##################################3
    m_num = Mn_num(8, t)
    ## map to standardized moments
    for i in range(len(m_num)):
        m_num[i] = m_num[i] / (st ** (i + 1.0))

    stt = str(t) + "  "
    for kk in range(len(m_num)):
        stt += str(m_num[kk]) + "  "
    stt += "\n"
    #gp.write(stt);

    plt.legend(frameon=False, bbox_to_anchor=(1.02, 1.02),  ncol=1)
    plt.xlim(-2.5, 2.5)
    ax.set_ylabel(r"$f(v|\hat{t}=\hat{t}_"+str(k)+")$")
    ax.set_xlabel(r"$v$")
    fig.set_size_inches(size * cm, size * cm)
    plt.savefig("TestCase_Boltz/t_"+str(k)+".pdf", format='pdf', bbox_inches="tight", dpi=300);
    ax.legend();
#plt.show()

gp.close()

'''
fp = open("Boltz_dimY_"+str(4+2*(model_num)), "w")
st = ""
for j in range(len(tt)):
    st+=str(tt[j])+"  "+str(DKL_val[j])+"\n"
fp.write(st);
fp.close()
'''