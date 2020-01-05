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
size = 6;
lw = 1
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})
colors = ['r', 'b', 'g', 'm', 'c', 'c', 'c', 'c']
markers = ["<","s","o",">","o", "o","o","o","o" ]

### read GPR models
import gpflow as gp
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import tensorflow as tf
'''
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
'''
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

#data_address = ["data/4l.txt", "data/6l.txt", "data/8l.txt"]
#model_names = ["models/4l_1000.txt", "models/6ln_1000.txt", "models/8ln_1000.txt"]
model_names = []
models = []
data_address = []
NN = [125,250,500,1000]#,200,300,400,500,1000]
#NN = [900]

mean_input = np.genfromtxt("data_test/GPR_mean_input.dat", dtype=np.float64)
deviation_input = np.genfromtxt("data_test/GPR_variance_input.dat", dtype=np.float64)
mean_output = np.genfromtxt("data_test/GPR_mean_output.dat", dtype=np.float64)
deviation_output = np.genfromtxt("data_test/GPR_variance_output.dat", dtype=np.float64)

for j in range(len(NN)):
    N = NN[j]
    print(str(N))
    data_address.append("data_test/4_10_N_4000.txt")
    model_names.append("data_test/4l_" + str(N) + ".md")
    m_name = model_names[j]
    models.append(gp.saver.Saver().load(m_name))
Np = 50

Error = np.zeros(len(model_names))
VarError = np.zeros(len(model_names))
Num_nan = np.zeros(len(model_names))
vval = [0.04,0.02,0.01]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
fig3, ax3 = plt.subplots();
for jj in range(len(vval)):
    val = vval[jj]
    ep = val#+np.random.rand(Np)*0.03
    qs = np.arange(-0.5, 0.5+1.0/Np, 1.0/Np)
    rs = (qs**2+1.0)*(1.0+ep)

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
        mmY = mm[dimX:]
        varY = vvar[dimX:]

        error = np.zeros(len(qs));
        vvv = np.zeros(len(qs));
        num_nan = 0;
        for ii in range(len(qs)):
            q = np.array([[qs[ii],rs[ii]]])
            # scale input
            x_transfered = transf_to(q, mean_input, deviation_input, 0, dimY - 2);
            output, err_output = model.predict_y(x_transfered)
            l = transf_back(output, mean_output, deviation_output, 0, dimY);
            l = l[0]
            I, d = integrate.quad(Z, -1e1, 1e1, args=(l, dimY));
            q_pred = Mom(l, dimY, dimY)
            er = np.linalg.norm(q_pred-np.array([0.0,1.0,qs[ii],rs[ii]]))/np.linalg.norm(q_pred)
            if np.isnan(er) == 0 and np.isinf(er) == 0:
                error[ii] = er;
                vvv[ii] = np.linalg.norm(err_output)
            else:
                num_nan = num_nan+1;
            Error[j] = np.mean(error);
            VarError[j] = np.mean(vvv);
    ax1.plot(NN, Error, color=colors[jj],marker=markers[jj],markersize=3, label=r"$d_"+str(jj)+"$")
    ax2.plot(NN, VarError, color=colors[jj],marker=markers[jj],markersize=3,linestyle="--")

    ax3.plot(qs, rs, color=colors[jj], linewidth=1,label=r"$d_"+str(jj)+"$",marker=markers[jj],markersize=3, markevery=3)


ax1.set_yscale("log")
ax2.set_yscale("log")
ax1.set_xscale("log")
ax1.set_ylim([1e-4, 1e0])
ax2.set_ylim([1e-4, 1e0])
ax1.set_xticks([100,1000])
ax2.set_xticks([100,1000])
ax1.legend(frameon=False,loc="lower left")
ax1.set_ylabel(r"$\mathrm{E}[{||p^{\mathrm{est}}-p^{\mathrm{in}}||_2}/{||p^{\mathrm{in}}||_2}]$")
ax2.set_ylabel(r"$\mathrm{E}[{|| \mathrm{Var}(\lambda^{\mathrm{est}}) ||_2}]$")
ax1.set_xlabel(r"$M$")
name = "data_test/error"


ax3.plot(qs, qs**2+1.0, color="black",linestyle="-")
ax3.fill_between(qs, 0.98, qs**2+1.0, color='lightgrey')
ax3.set_xlim([-0.5, 0.5])
ax3.set_ylim([0.98, 1.3])
ax3.set_xlabel(r"$p_3$")
ax3.set_ylabel(r"$p_4$")
ax3.legend(frameon=False,loc="upper center")
name2 = "data_test/q_r"


plt.tight_layout();

fig.set_size_inches(1.12*size * cm,1.12* size * cm)
fig.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
fig3.set_size_inches(size * cm, size * cm)
fig3.savefig(name2 + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
#plt.show();
'''
fp = open("Boltz_dimY_"+str(4+2*(model_num)), "w")
st = ""
for j in range(len(tt)):
    st+=str(tt[j])+"  "+str(DKL_val[j])+"\n"
fp.write(st);
fp.close()
'''