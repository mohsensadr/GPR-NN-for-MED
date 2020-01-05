import pickle
import numpy as np
import gpflow as gp
from scipy import integrate
#from integration_max_entropy import moments
#from sampling import samples
from random import randint
from numpy import linalg as LA
from numpy.linalg import inv
import time
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import tensorflow as tf

def transf_to(xx,m,v,i1,i2):
    xxnew = xx.copy();
    for i in range(i1,i2):
        if abs(v[i]) > 1e-15:
            xxnew[:,i] = (xx[:,i] - m[i] ) / v[i] ** 0.5+1.0
    return xxnew

def transf_back(xx,m,v,i1,i2):
    xxnew = xx.copy();
    for i in range(i1,i2):
        if abs(v[i])>1e-15:
            xxnew[:, i] = (xx[:,i]-1.0)* v[i] ** 0.5 + m[i]
    return xxnew

def f1(x,mu,sig):
    return 1.0/np.sqrt(2.0*np.pi*sig**2)*np.exp(- (x-mu)**2/(2.0*sig**2));

def f(x,mu1,sig1,mu2,sig2):
    return 0.5*(f1(x,mu1,sig1)+f1(x,mu2,sig2))
def xif(x,mu1,sig1,mu2,sig2,i):
    return x**i*f(x,mu1,sig1,mu2,sig2)

def ft(x,mu1,sig1,mu2,sig2,t,nu):
    emnt = np.exp(-nu*t)
    return f(x,mu1,sig1,mu2,sig2)*emnt+(1-emnt)*f1(x,0.0,1.0);

def xif1(x,mu,sig,i):
    return x**i*f1(x,mu,sig)

def f3(x,mu1,sig1,mu2,sig2,mu3,sig3):
    return (1.0/3.0)*(f1(x,mu1,sig1)+f1(x,mu2,sig2)+f1(x,mu3,sig3))
def xif3(x,mu1,sig1,mu2,sig2,mu3,sig3,i):
    return x**i*f3(x,mu1,sig1,mu2,sig2,mu3,sig3)

'''
def Z(v, l):
    l1 = l
    if len(l1)==4:
        return np.exp(-v * l1[0] - v**2 * l1[1] - v**3 * l1[2] - v**4 * l1[3])
    elif len(l1)==6:
        return np.exp(-v * l1[0] - v ** 2 * l1[1] - v ** 3 * l1[2] - v ** 4 * l1[3] - v ** 5 * l1[4] - v ** 6 * l1[5])
    elif len(l1)==8:
        return np.exp(-v * l1[0] - v**2 * l1[1] - v**3 * l1[2] - v**4 * l1[3] - v**5 * l1[4] - v**6 * l1[5] - v**7 * l1[6] - v**8 * l1[7])
'''
def logpq(x, l, mu1, sig1, mu2, sig2, it):
    l1 = l[0]
    y = np.log(0.5/np.sqrt(2.0*np.pi*sig1**2)) - (x-mu1)**2/(2.0*sig1**2) + np.log(0.5/np.sqrt(2.0*np.pi*sig2**2)) - (x-mu2)**2/(2.0*sig2**2)
    for i in range(len(l1)):
        y = y - x**(i+1) * l1[i];
    y = y #+ np.log(it)
    return y;
def DKL(x, l, mu1, sig1, mu2, sig2, it):
    return f(x,mu1,sig1,mu2,sig2)*logpq(x, l, mu1, sig1, mu2, sig2,it)

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


modes = 2;

if modes == 2:
    mu = [0.7, 0.8, 0.9, 0.95, 0.98]
    sig = [0.55, 0.53, 0.43, 0.284, 0.21]

    mu = [0.8, 0.9, 0.95, 0.98]
    sig = [0.7, 0.4, 0.31, 0.2]

    mu = [0.98]
    sig = [0.2]

    #mu = [0.8]
    #sig = [0.7]

    #mu = [0.98]
    #sig = [0.2]
elif modes == 3:
    mu = [0.8, 1.0, 1.205]
    sig = [0.2, 0.2, 0.21]
transient = True;

N = 6;
T = 20.0;
if N>1:
    dt = T/(N-1.0);
else:
    dt = 0
nu = 0.25;
lamb = ["4", "6", "8"]
method = "GPR"

stat = "var"
#model_names = ["models/4_var_5e-2_1000.txt", "models/6_var_1e-4_1000.txt", "models/8_var_1e-7_1000.txt"]
#data_address = ["data/4l5e-2_var4th.txt", "data/6l1e-4_var6th.txt", "data/8l1e-7_var8th.txt"]
data_address = ["data/4l.txt", "data/6l.txt", "data/8l.txt"]
model_names = ["models/4ln_1000.txt", "models/6ln_1000.txt", "models/8ln_1000.txt"]

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import numpy as np
from scipy import integrate

def my_integral(y,x):
    n = len(x);
    dx = (x[-1]-x[0])/(1.0*n)
    sd = 0.0;
    for i in range(n):
        sd += y[i]*dx;
    return sd;
def predicting(q, mmX, varX, dimX, model, mmY, varY):
    # scale input
    q_sc = transf_to(q, mmX, varX, 2, dimX + 2);
    # predict
    ystar2, varstar2 = model.predict_y(q_sc[:, 2:dimX + 2])
    print("varstar = "+str(varstar2))
    # transfer back
    q_tf = transf_back(q_sc, mmX, varX, 2, dimX + 2);
    la_tf = transf_back(ystar2, mmY, varY, 0, dimY);
    #if dimY != 4 and dimY != 6:
    #    la_tf = np.append(la_tf[0], x0[20 + dimY, 0]);
    #else:
    la_tf = la_tf[0]
    #else:
    #    la_tf = np.append(la_tf[0], x0[20 + dimY, 0]);
    return la_tf, varstar2[0]

msize = 3
size = 4.5;
size = 6;
lw = 1
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})
colors = ['black', 'r', 'b', 'g', 'm', 'c', 'c', 'c', 'c']
linestyles = [':', '--', '-']
markers = ["<","s","o",">","o", "o","o","o","o" ]
xx = np.arange(-10.0, 10.0, 0.01)
from cycler import cycler
models = []
DKL_val = []
[DKL_val.append([]) for _ in range(len(mu))]
for j in range(len(model_names)):
    m_name = model_names[j]
    models.append(gp.saver.Saver().load(m_name))
for case in range(len(mu)):
    La = [];
    intt = []
    [intt.append([]) for _ in range(len(model_names))]
    [La.append([]) for _ in range(len(model_names))]
    qdiff_all = [];
    [qdiff_all.append([]) for _ in range(len(model_names))]

    mu1 = mu[case]  # + ((mu_end-mu_beg)/(n-1))*j
    mu2 = -mu1;
    sig1 = sig[case];
    sig2 = np.sqrt(2.0 - (sig1 ** 2 + 2 * mu1 ** 2))

    ff = f(xx, mu1, sig1, mu2, sig2);



    tt = [0.0, T*0.15, T*0.4, T]
    #tt = [T*0.4]
    dt = T/len(tt)


    qt = []
    [qt.append([]) for _ in range(len(tt))]
    gt = []
    [gt.append([]) for _ in range(len(tt))]
    k=0
    fig2, ax2 = plt.subplots()
    for jj in range(len(tt)):
        kkl = np.zeros(len(model_names))
        t = tt[jj]
        fti = ft(xx, mu1, sig1, mu2, sig2, t, nu);
        fig, ax = plt.subplots()
        for j in range(len(model_names)):
            address = data_address[j]
            x0 = np.loadtxt(address, skiprows=1, unpack=True);
            dim = len(x0[:, 0]);  ## 26 here

            vvar = []
            mm = []
            for i in range(dim):
                mm.append(np.mean(x0[i, :]));
                vvar.append(np.var(x0[i, :]));
            model = models[j]  # gp.saver.Saver().load(m_name)

            N1 = len(model.X.value)
            dimX = len(model.X.value[0])
            dimY = len(model.Y.value[0]);

            if dimX == 2:
                meq = [0.0, 1.0, 0.0, 3.0]
            elif dimX == 4:
                meq = [0.0, 1.0, 0.0, 3.0, 0.0, 15.0]
            elif dimX == 6:
                meq = [0.0, 1.0, 0.0, 3.0, 0.0, 15.0, 0.0, 105.0]
            elif dimX == 8:
                meq = [0.0, 1.0, 0.0, 3.0, 0.0, 15.0, 0.0, 105.0, 0.0, 945.0]
            meq = np.array(meq);
            mmX = mm[0:dimX + 2]
            varX = vvar[0:dimX + 2]
            mmY = mm[20:]
            varY = vvar[20:]

            q = [[]];
            for i in range(0, dimX + 2):
                if modes == 2:
                    I, d2 = integrate.quad(xif, -1e1, 1e1, args=(mu1, sig1, mu2, sig2, i + 1));
                elif modes == 3:
                    I, d2 = integrate.quad(xif3, -1e1, 1e1, args=(mu1, sig1, mu2, sig2, mu3, sig3, i + 1));
                q[0].append(I);
            q = np.array(q)


            m = [[]];
            m[0].append( np.exp(-nu*t)*q[0]+(1.0-np.exp(-nu*t))*meq )
            #print("i="+str(i)+" t="+str(t)+" m[0] = "+str(m[0])+" where q[0]="+str(q[0])+"and meq"+str(meq))
            m = np.array(m)

            '''
            if dimY==8:
                sttt = ""
                for kkk in range(dimY):
                    sttt += str(m[0][0][kkk])+"   "
                qp = open("BGK/points_to_add.txt", "a")
                sttt += "\n"
                qp.write(sttt);
                qp.close()
            '''
            start = time.time()
            l, varst = predicting(m[0], mmX, varX, dimX, model, mmY, varY);
            end = time.time()
            itt, d = integrate.quad(Z, -1e1, 1e1, args=(l,dimY));
            #if varst[0] < 1e20:
            zz = Z(xx, l,dimY) / itt
            #plt.plot(xx, zz, color=colors[k],marker="o", markevery=40,markersize=3,label=r"$f^\lambda_"+lamb[j]+"(t="+'{:.0f}'.format(float(t))+")$",linewidth=0.9)
            plt.plot(xx, zz, color=colors[j+1], marker=markers[j], markevery=50+10*j, markersize=msize,linestyle="-",
                     label=r"$f^\lambda_" + lamb[j] + "$", linewidth=0.9)
            pp = Mom(l,dimY,dimY)


            sttt = str(t)+"   "
            for kkk in range(dimY):
                sttt += str(pp[kkk]) + "   "
            for kkk in range(dimY):
                sttt += str(l[kkk]) + "   "
            sttt += str(end-start)+"\n"
            qp = open("BGK/BGK_dimY_"+str(dimY)+".txt", "a")
            sttt += "\n"
            qp.write(sttt);
            qp.close()


            kkl[j] =  kl(fti,zz)


        ax2.plot(lamb,kkl,label=r"$t_"+str(jj)+"$", marker=markers[jj], markersize=3)
        ax2.set_ylabel(r"$D_{KL}(f^{\mathrm{BGK}}||f^\lambda_N)$")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"$N$")



        #plt.plot(xx, fti, color=colors[k],marker="s", markevery=40,markersize=3,label=r"$f^{\mathrm{ex.}}(t="+'{:.0f}'.format(float(t))+")$",linewidth=0.9)
        ax.plot(xx, fti, color="black",linestyle='-',
                     label=r"$f^{\mathrm{BGK}}$", linewidth=0.9)


        ax.set_xlim([-3, 3])
        #ax.set_ylim([-0.1, 1.5])
        ax.set_ylabel(r"$f(v|t=t_"+str(jj)+")$")
        ax.set_xlabel(r"$v$")
        #plt.legend(frameon=False, bbox_to_anchor=(1.05, 0.09), ncol=1)
        #plt.legend(bbox_to_anchor=(1.05, 1.05), loc=2, borderaxespad=0.,frameon=False)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.02), ncol=1)
        st_mu = str(mu1) + "_" + str(sig1)
        st_mu = st_mu.replace(".", "")
        fig.set_size_inches(size * cm, size * cm)
        name = "BGK/evolution_t" + str(jj) + "_mean098"+stat
        fig.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
        #plt.show()
        #plt.show()
        #I, d = integrate.quad(Z, -1e1, 1e1, args=(La[j]));
        #intt[j].append(I);
        #print("intt[j] = "+str(intt[j]))

        #I, d = integrate.quad(DKL, -1e1, 1e1, args=(La[j], mu1, sig1, mu2, sig2, intt[j]));
        #DKL_val[case].append(I);


ax2.set_ylim([1e-5, 1])
ax2.legend(frameon=False, bbox_to_anchor=(1.02, 1.02), ncol=1)
st_mu = "BGK"
fig2.set_size_inches(size * cm, size * cm)
name = "BGK/KL_t_N_mean098"+stat
fig2.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
#plt.show()

if method == "GPR":
    if transient == False:
        dl = [4,6,8]
    else:
        dl = [1]## TO DO
if method == "SVGP":
    dl = MM;

mm = range(len(qt[0]));
'''
fig, ax = plt.subplots();
[plt.plot(dl, DKL_val[j], label=r"$\mu_1 = "+str(mu[j])+", \sigma_1 = "+str(sig[j])+"$", marker="o",markersize=6) for j in range(len(mu))]
plt.legend(frameon=False,loc="lower left")
ax.set_ylabel(r"$D_{KL}(f_{\mathrm{ex}}||f_\lambda^{(n)})$")
ax.set_yscale("log")
if method == "GPR":
    ax.set_xlabel(r"$n$ (degree of $v^j \lambda_j$)")
    name = "TestCase1/"+str(modes)+"modes_kl_468l_" + "dist"
    ax.set_ylim([1e-4, 1.0])
elif method == "SVGP":
    ax.set_xlabel(r"$M$")
    name = "TestCase1/SVGP"
    #for j in range(len(MM)):
    #            name += str(MM[j])
    name += "_" + "DKL"
    ax.set_ylim([1e-2, 1e1])
plt.xticks(dl)
fig.set_size_inches(size * cm, size * cm)
plt.legend(frameon=False, loc='upper right')
#plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
print(DKL_val)
'''


'''
fig, ax = plt.subplots();
#qt = np.array(qt);
#gt = np.array(gt);

[plt.plot(mm[2:-1],abs(np.array(qt[i][2:-1])-np.array(gt[i][2:-1]))/np.array(qt[i][2:-1]),label="t="+'{:.0f}'.format(float(tt[i])),marker="o",markersize=5,
                  color=colors[i], linewidth=lw) for i in range(len(tt))]
ax.set_yscale("log")
plt.xticks(mm[2:-1])
plt.legend(frameon=False, loc='lower right')
ax.set_ylabel(r"$|p_i-p^{\mathrm{Exact}}_i|$")
ax.set_xlabel(r"$i$")
fig.set_size_inches(size * cm, size * cm)
name = "transient/error_"+str(dimY)+"_mean09"
plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
#plt.show();
'''

size = 10;
fig, ax = plt.subplots();

'''
i1=2;i2=-1
ii = [2,4,6,8]
for j in range(len(qdiff_all)):
    [plt.plot(mm[2::2], qdiff_all[j][i][2::2],
              label=r"$f^\lambda_"+lamb[j]+"(t_"+str(i)+")$", marker=markers[j], markersize=5,linestyle=linestyles[j],
              color=colors[i], linewidth=0.9) for i in range(len(tt))]
ax.set_yscale("log")
plt.xticks(mm[2::2])
ax.set_ylabel(r"$|p_i-p^{\mathrm{ex}}_i|/|p^{\mathrm{ex}}_i|$")
ax.set_xlabel(r"$i$")
#plt.legend(frameon=False, bbox_to_anchor=(1.05, 1.15), ncol=1)
#plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.,frameon=False,ncol=2)
plt.legend(bbox_to_anchor=(1.05, 1.05), loc=2, borderaxespad=0.,frameon=False)
fig.set_size_inches(size * cm, size * cm)
name = "transient/error_mean098"+stat
plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
plt.show();
'''