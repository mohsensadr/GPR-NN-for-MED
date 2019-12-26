import pickle
import numpy as np
import gpflow as gp
from scipy import integrate
#from integration_max_entropy import moments
#from sampling import samples
from random import randint
from numpy import linalg as LA
from numpy.linalg import inv

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

def f1(x,mu,sig):
    return 1.0/np.sqrt(2.0*np.pi*sig**2)*np.exp(- (x-mu)**2/(2.0*sig**2));

def f(x,mu1,sig1,mu2,sig2):
    return 0.5*(f1(x,mu1,sig1)+f1(x,mu2,sig2))
def xif(x,mu1,sig1,mu2,sig2,i):
    return x**i*f(x,mu1,sig1,mu2,sig2)

def xif1(x,mu,sig,i):
    return x**i*f1(x,mu,sig)

def f3(x,mu1,sig1,mu2,sig2,mu3,sig3):
    return (1.0/3.0)*(f1(x,mu1,sig1)+f1(x,mu2,sig2)+f1(x,mu3,sig3))
def xif3(x,mu1,sig1,mu2,sig2,mu3,sig3,i):
    return x**i*f3(x,mu1,sig1,mu2,sig2,mu3,sig3)

def Z(v, l):
    l1 = l[0]
    if len(l1)==4:
        return np.exp(-v * l1[0] - v**2 * l1[1] - v**3 * l1[2] - v**4 * l1[3])
    elif len(l1)==6:
        return np.exp(-v * l1[0] - v ** 2 * l1[1] - v ** 3 * l1[2] - v ** 4 * l1[3] - v ** 5 * l1[4] - v ** 6 * l1[5])
    elif len(l1)==8:
        return np.exp(-v * l1[0] - v**2 * l1[1] - v**3 * l1[2] - v**4 * l1[3] - v**5 * l1[4] - v**6 * l1[5] - v**7 * l1[6] - v**8 * l1[7])

def logpq(x, l, mu1, sig1, mu2, sig2, it):
    l1 = l[0]
    y = np.log(0.5/np.sqrt(2.0*np.pi*sig1**2)) - (x-mu1)**2/(2.0*sig1**2) + np.log(0.5/np.sqrt(2.0*np.pi*sig2**2)) - (x-mu2)**2/(2.0*sig2**2)
    for i in range(len(l1)):
        y = y - x**(i+1) * l1[i];
    y = y #+ np.log(it)
    return y;
def DKL(x, l, mu1, sig1, mu2, sig2, it):
    return f(x,mu1,sig1,mu2,sig2)*logpq(x, l, mu1, sig1, mu2, sig2,it)

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

    mu = [0.8, 0.9, 0.95]
    sig = [0.3, 0.2, 0.15]

    #mu = [0.9]
    #sig = [0.4]

    #mu = [0.8]
    #sig = [0.7]

    #mu = [0.98]
    #sig = [0.2]
elif modes == 3:
    mu = [0.8, 1.0, 1.205]
    sig = [0.3, 0.2, 0.21]

method = "GPR"
#method = "SVGP"
if method == "GPR":
    model_names = ["models/4l_GPR_RBF_womN1_2000.txt","models/6l_GPR_RBF_womN1_3000.txt","models/8l_GPR_RBF_womN1_4000.txt"]
    model_names = ["models_old/4l_GPR_RBF_womN1_2000.txt", "models_old/6l_new.txt",
                   "models_old/8l_GPR_RBF_womN1_4000.txt"]
    data_address = ["myfile_4l.txt","myfilel6_new.txt","myfile_8l4.txt"]

    stat = "var"
    data_address = ["data/4l5e-2_"+stat+"4th.txt", "data/6l1e-4_"+stat+"6th.txt", "data/8l1e-7_"+stat+"8th.txt"]
    model_names = ["models/4_"+stat+"_5e-2_1000.txt", "models/6_"+stat+"_1e-4_1000.txt", "models/8_"+stat+"_1e-7_1000.txt"]

    data_address = ["data/4l.txt", "data/6l.txt","data/8l.txt"]
    model_names = ["models/4l_1000.txt", "models/6l_1000.txt", "models/8l_1000.txt"]

elif method =="SVGP":
    # 20 works
    MM = [10,20, 30]#, 32, 64, 128]
    model_names = []; data_address = [];
    for j in range(len(MM)):
        #model_names.append("models/testing_new_SVGP_M"+str(MM[j])+".txt")
        model_names.append("models/SGPR_M" + str(MM[j]) + ".txt")
        if j ==-1 :
            data_address.append("myfile_8l4.txt")
        else:
            data_address.append("myfilel6.txt");
#    model_names = ["models/testing_new_SVGP_M"+str(MM[0])+".txt","models/testing_new_SVGP_M"+str(MM[1])+".txt","models/testing_new_SVGP_M"+str(MM[2])+".txt"]
#    data_address = ["myfilel6.txt","myfilel6.txt","myfilel6.txt"]
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

lla = ["(a)","(b)","(c)"]
msize = 0
size = 6;
lw = 1
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})
colors = ['black', 'r', 'b', 'g', 'm', 'c']
markers = ["<","s","o"]
xx = np.arange(-10.0, 10.0, 0.0001)
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
    if modes==2:
        mu1 = mu[case]  # + ((mu_end-mu_beg)/(n-1))*j
        mu2 = -mu1;
        sig1 = sig[case];
        sig2 = np.sqrt(2.0 - (sig1 ** 2 + 2 * mu1 ** 2))

        ff = f(xx, mu1, sig1, mu2, sig2);
    elif modes ==3:
        mu1 = mu[case]  ## 0.8 1.0 1.2
        sig1 = sig[case];
        mu2 = -mu1;
        mu3 = -(mu1 + mu2);
        sig2 = sig1
        sig3 = np.sqrt(3.0 - (sig1 ** 2 + sig2 ** 2 + mu1 ** 2 + mu2 ** 2 + mu3 ** 2))
        ff = f3(xx, mu1, sig1, mu2, sig2, mu3, sig3);

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

        q = [[]];
        for i in range(0, dimX + 2):
            if modes == 2:
                I, d2 = integrate.quad(xif, -1e1, 1e1, args=(mu1, sig1, mu2, sig2, i + 1));
            elif modes==3:
                I, d2 = integrate.quad(xif3, -1e1, 1e1, args=(mu1, sig1, mu2, sig2,mu3, sig3, i + 1));
            q[0].append(I);
        q = np.array(q)
        # scale input
        q_sc = transf_to(q, mmX, varX, 2, dimX + 2);
        # predict
        ystar2, varstar2 = model.predict_y(q_sc[:, 2:dimX + 2])
        print("mu1=" + str(mu1) + " var = " + str(varstar2[0, 0]))
        # transfer back
        q_tf = transf_back(q_sc, mmX, varX, 2, dimX + 2);
        la_tf = transf_back(ystar2, mmY, varY, 0, dimY);
        #if dimY==4:
        la_tf = la_tf[0]
        #else:
        #    la_tf = np.append(la_tf[0], x0[20+dimY, 0]);
        La[j].append(la_tf);
        I, d = integrate.quad(Z, -1e1, 1e1, args=(La[j]));
        intt[j].append(I);
        print("intt[j] = "+str(intt[j]))

        #I, d = integrate.quad(DKL, -1e1, 1e1, args=(La[j], mu1, sig1, mu2, sig2, intt[j]));
        #DKL_val[case].append(I);



        fl = Z(xx, La[j]) / intt[j];
        DKL_val[case].append(kl(ff, fl));


    fig, ax = plt.subplots();
    if modes == 2:
        plt.plot(xx, f(xx, mu1, sig1, mu2, sig2), label=r"$f^{\mathrm{bi}}$",
              marker="o", markevery=40, linestyle="-", markersize=msize, color=colors[0],linewidth=lw)
    elif modes ==3:
        plt.plot(xx, f3(xx, mu1, sig1, mu2, sig2, mu3, sig3), label=r"$f_{\mathrm{bi}}$",
              marker="o", markevery=40, linestyle="-", markersize=msize, color=colors[0],linewidth=lw)
    if method == "GPR":
        [plt.plot(xx, Z(xx, La[j]) / intt[j], marker=markers[j], markevery=4000+j*100, label=r"$f^{\lambda}_{" + str(2*j+4) + "}$",
                  markersize=3, color=colors[j + 1], linewidth=lw) for j in range(len(model_names))]
        #[label.append("$f_{\lambda}^{(" + str(2*j+4) + ")}$") for j in range(len(model_names))];
    elif method == "SVGP":
        [plt.plot(xx, Z(xx, La[j]) / intt[j], marker="s", markevery=40, label=r"$f_{\lambda,M_"+str(j+1)+"}$",
                  markersize=msize, color=colors[j + 1], linewidth=lw) for j in range(len(MM))]
        #[label.append(r"$f_{\lambda}^{(M="+str(MM[j])+")}$") for j in range(len(MM)) ];
    #[plt.plot(xx, Z(xx, La[j]) / intt[j], marker="s", markevery=40,
    #          markersize=msize, color=colors[j+1],linewidth=lw) for j in range(len(model_names))]
    ax.set_xlim([-3.0, 3.0])
    #ax.set_ylim([-0.1, 1.5])
    ax.set_ylabel(r"$f(x)$")
    ax.set_xlabel(r"$x$")
    plt.legend(frameon=False, bbox_to_anchor=(1.02, 1.02),  ncol=1)

    plt.text(0.1, 0.9, lla[case], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    st_mu = str(mu1)+"_"+str(sig1)
    st_mu = st_mu.replace(".", "")
    if method == "GPR":
        name = "TestCase1/"+str(modes)+"modes_4l_6l_8l_" + "dist_" + st_mu+stat
    elif method == "SVGP":

        name = "TestCase1/SVGP"
        #for j in range(len(MM)):
        #    name += str(MM[j])
        name += "_" + "dist_" + st_mu
    fig.set_size_inches(size * cm, size * cm)
    plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
    #plt.show()

fig, ax = plt.subplots();
if method == "GPR":
    dl = [4,6,8]
if method == "SVGP":
    dl = MM;

[plt.plot(dl, DKL_val[j], label=lla[j], marker=markers[j],markersize=3) for j in range(len(mu))]
plt.legend(frameon=False,loc="lower left")
ax.set_ylabel(r"$D_{KL}(f^{\mathrm{bi}}||f^\lambda_N)$")
ax.set_yscale("log")
if method == "GPR":
    ax.set_xlabel(r"$N$")
    name = "TestCase1/"+str(modes)+"modes_kl_468l_" + "dist"+stat
    ax.set_ylim([1e-3, 1.0])
elif method == "SVGP":
    ax.set_xlabel(r"$M$")
    name = "TestCase1/SVGP"
    #for j in range(len(MM)):
    #            name += str(MM[j])
    name += "_" + "DKL"
    ax.set_ylim([1e-2, 1e1])
plt.xticks(dl)
fig.set_size_inches(size * cm, size * cm)
plt.legend(frameon=False, loc='lower left')
plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
plt.show();
print(DKL_val)
