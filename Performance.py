import gpflow as gp
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import tensorflow as tf
import numpy as np
import time
from scipy import integrate

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



import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
size = 6;
lw = 1
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})
colors = ['black', 'r', 'b', 'g', 'm', 'c']
markers = ["<","s","o"]

##############################################################################






T=20
tt = [0.0, T*0.15, T*0.4, T]

fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()

fig2 = plt.figure()
ax3 = fig2.add_subplot(111)

#fig3 = plt.figure()
#ax4 = fig3.add_subplot(111)

dimy = [4,6,8]
for j in range(len(dimy)):
    dimY = dimy[j]

    cost = []; [cost.append([]) for _ in range(len(tt))];
    err_l = []; [err_l.append([]) for _ in range(len(tt))];
    err_p = [];[err_p.append([]) for _ in range(len(tt))];
    varr = [];[varr.append([]) for _ in range(len(tt))];

    x = np.loadtxt("BGK/BGK_dimY_" + str(dimY) + ".txt", skiprows=0, unpack=True);
    y = np.loadtxt("BGK/BGK_dimY_" + str(dimY) + "_direct.txt", skiprows=0, unpack=True);

    for case in range(len(tt)):
        pl = x[1:dimY + 1, case]
        ll = x[dimY + 1:dimY * 2 + 1, case]
        tl = x[-1, case]

        p_ex = y[1:dimY+1,case]
        l_ex = y[dimY+1:dimY*2+1,case]
        t_ex = y[-1,case]

        cost[case] = tl/t_ex;
        err_l[case] = np.linalg.norm(l_ex-ll)/np.linalg.norm(l_ex)
        err_p[case] = np.linalg.norm(p_ex - pl) / np.linalg.norm(p_ex)
        #varr[case] = varstar2[0, 0]
    ax1.plot(tt, err_l, label=r"$N=" + str(dimY) + "$", marker=markers[j],
             markersize=3, color=colors[j + 1], linewidth=lw)
    #ax2.plot(tt, err_p, label=r"$N=" + str(dimY) + "$", marker=markers[j],
    #         markersize=3, color=colors[j + 1], linewidth=lw,linestyle="-.")
    ax3.plot(tt, cost, label=r"$N=" + str(dimY) + "$", marker=markers[j],
             markersize=3, color=colors[j + 1], linewidth=lw, linestyle="-")
    #ax4.plot(cases, varr, label=r"$N=" + str(dimY) + "$", marker=markers[j],
    #         markersize=3, color=colors[j + 1], linewidth=lw, linestyle="-")
ax1.set_yscale("log")
#ax2.set_yscale("log")
ax3.set_yscale("log")
#ax4.set_yscale("log")

ax1.set_ylim([1e-8, 1e-2])
#ax2.set_ylim([5e-8, 1e4])
#ax4.set_ylim([1e-7, 1e1])
ax1.legend(frameon=False, loc='lower right')

ax1.set_ylabel(r"$||\lambda^{\mathrm{ex}}-\lambda^{\mathrm{est}}||_2/||\lambda^{\mathrm{ex}}||_2$")
#ax2.set_ylabel(r"$||p^{\mathrm{ex}}-p^{\mathrm{est}}||_2/||p^{\mathrm{ex}}||_2$")
ax1.set_xlabel(r"$t$")

ax3.legend(frameon=False, loc='lower right')
ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$\tau_{\mathrm{GPR}}^{\mathrm{pred}}/\tau_{\mathrm{direct}}$")

#ax4.legend(frameon=False, loc='upper right')
#ax4.set_xlabel("Test Case")
#ax4.set_ylabel(r"$||\mathrm{Var}(\lambda^{\mathrm{est}})||_2$")

plt.tight_layout();

fig.set_size_inches(size * cm, size * cm)
fig.savefig("BGK/error.pdf", format='pdf', bbox_inches="tight", dpi=300);
fig2.set_size_inches(size * cm, size * cm)
fig2.savefig("BGK/t.pdf", format='pdf', bbox_inches="tight", dpi=300);
#fig3.set_size_inches(size * cm, size * cm)
#fig3.savefig("BGK/varr.pdf", format='pdf', bbox_inches="tight", dpi=300);





##############################################################################

mu = [0.8, 0.9, 0.95]
sig = [0.3, 0.2, 0.15]
cases = ["(a)", "(b)", "(c)"]
data_address = ["data/4l.txt", "data/6l.txt", "data/8l.txt"]
model_names = ["models/4l_1000.txt", "models/6ln_1000.txt", "models/8ln_1000.txt"]

models = []
for j in range(len(model_names)):
    m_name = model_names[j]
    models.append(gp.saver.Saver().load(m_name))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

fig2 = plt.figure()
ax3 = fig2.add_subplot(111)

fig3 = plt.figure()
ax4 = fig3.add_subplot(111)

dimy = [4,6,8]
for j in range(len(dimy)):
    dimY = dimy[j]

    cost = []; [cost.append([]) for _ in range(len(mu))];
    err_l = []; [err_l.append([]) for _ in range(len(mu))];
    err_p = [];[err_p.append([]) for _ in range(len(mu))];
    varr = [];[varr.append([]) for _ in range(len(mu))];
    for case in range(len(mu)):
        mu1 = mu[case]  # + ((mu_end-mu_beg)/(n-1))*j
        mu2 = -mu1
        sig1 = sig[case]
        sig2 = np.sqrt(2.0 - (sig1 ** 2 + 2 * mu1 ** 2))

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

        mmX = mm[0:dimX + 2]
        varX = vvar[0:dimX + 2]
        mmY = mm[20:]
        varY = vvar[20:]

        q = [[]];
        for i in range(0, dimX + 2):
            I, d2 = integrate.quad(xif, -1e1, 1e1, args=(mu1, sig1, mu2, sig2, i + 1));
            q[0].append(I);
        q = np.array(q)
        # scale input
        q_sc = transf_to(q, mmX, varX, 2, dimX + 2);
        # predict
        start_time = time.time()
        ystar2, varstar2 = model.predict_y(q_sc[:, 2:dimX + 2])
        print("mu1=" + str(mu1) + " var = " + str(varstar2[0, 0]))
        # transfer back
        q_tf = transf_back(q_sc, mmX, varX, 2, dimX + 2);
        la_tf = transf_back(ystar2, mmY, varY, 0, dimY);
        # if dimY==4:
        la_tf = la_tf[0]
        end_time = time.time()

        p = Mom(la_tf, dimY, dimY)

        y = np.loadtxt("../direct/dimY_"+str(dimY)+"_mu1_"+str(mu1), skiprows=0, unpack=True);
        p_ex = y[0:dimY]
        l_ex = y[dimY:dimY*2]
        t_ex = y[-1]

        cost[case] = (end_time-start_time)/t_ex;
        err_l[case] = np.linalg.norm(l_ex-la_tf)/np.linalg.norm(l_ex)
        err_p[case] = np.linalg.norm(p_ex - p) / np.linalg.norm(p_ex)
        varr[case] = varstar2[0, 0]
    ax1.plot(cases, err_l, label=r"$N=" + str(dimY) + "$", marker=markers[j],
             markersize=3, color=colors[j + 1], linewidth=lw)
    ax2.plot(cases, err_p, label=r"$N=" + str(dimY) + "$", marker=markers[j],
             markersize=3, color=colors[j + 1], linewidth=lw,linestyle="-.")
    ax3.plot(cases, cost, label=r"$N=" + str(dimY) + "$", marker=markers[j],
             markersize=3, color=colors[j + 1], linewidth=lw, linestyle="-")
    ax4.plot(cases, varr, label=r"$N=" + str(dimY) + "$", marker=markers[j],
             markersize=3, color=colors[j + 1], linewidth=lw, linestyle="-")
ax1.set_yscale("log")
ax2.set_yscale("log")
ax3.set_yscale("log")
ax4.set_yscale("log")

ax1.set_ylim([5e-8, 1e6])
ax2.set_ylim([5e-8, 1e6])
ax4.set_ylim([1e-7, 1e1])
ax1.legend(frameon=False, loc='upper left')

ax1.set_ylabel(r"$||\lambda^{\mathrm{ex}}-\lambda^{\mathrm{est}}||_2/||\lambda^{\mathrm{ex}}||_2$")
ax2.set_ylabel(r"$||p^{\mathrm{in}}-p^{\mathrm{est}}||_2/||p^{\mathrm{in}}||_2$")
ax1.set_xlabel("Test Case")

ax3.legend(frameon=False, loc='upper right')
ax3.set_xlabel("Test Case")
ax3.set_ylabel(r"$\tau_{\mathrm{GPR}}^{\mathrm{pred}}/\tau_{\mathrm{direct}}$")

ax4.legend(frameon=False, loc='upper left')
ax4.set_xlabel("Test Case")
ax4.set_ylabel(r"$||\mathrm{Var}(\lambda^{\mathrm{est}})||_2$")

plt.tight_layout();

fig.set_size_inches(size * cm, size * cm)
fig.savefig("TestCase1/error.pdf", format='pdf', bbox_inches="tight", dpi=300);
fig2.set_size_inches(size * cm, size * cm)
fig2.savefig("TestCase1/t.pdf", format='pdf', bbox_inches="tight", dpi=300);
fig3.set_size_inches(0.9*size * cm, 0.9*size * cm)
fig3.savefig("TestCase1/varr.pdf", format='pdf', bbox_inches="tight", dpi=300);
#ax.set_xscale("log")
#plt.show()


dimy = [4,6,8]
fig, ax = plt.subplots();
for j in range(len(dimy)):
    dimY = dimy[j]
    case = "MED_perf_dimY_"+str(dimY)+".txt"
    x = np.loadtxt(case, skiprows=0, unpack=True);
    if dimY==4:
        t0 = x[2][0]
    M = np.array(x[1]);
    t = np.array(x[2]);
    plt.plot(M, t/t0, label=r"$N="+str(dimY)+"$", marker=markers[j],
             markersize=3,color=colors[j + 1])
ax.set_yscale("log")
ax.set_xscale("log")
#ax.set_xlim([1e1, 1e4])
#ax.set_ylim([8e-1, 1e3])
ax.set_ylabel(r"$\tau^{\mathrm{tr}}/\tau^{\mathrm{tr}}_0$")
ax.set_xlabel(r"$M$")
fig.set_size_inches(size * cm, size * cm)
plt.legend(frameon=False, loc='upper left')
plt.savefig("Performance/MED_train_time.pdf", format='pdf', bbox_inches="tight", dpi=300);



dimy = [4,6,8]
fig, ax = plt.subplots();
for j in range(len(dimy)):
    dimY = dimy[j]
    case = "new_data_geb_perf_dimY_"+str(dimY)+".txt"
    x = np.loadtxt(case, skiprows=0, unpack=True);
    if dimY==4:
        t0 = x[2][0]
    M = np.array(x[1]);
    t = np.array(x[2]);
    plt.plot(M, t/t0, label=r"$N="+str(dimY)+"$", marker=markers[j],
             markersize=3,color=colors[j + 1])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\tau^{\mathrm{gen}}/\tau^{\mathrm{gen}}_0$")
ax.set_xlabel(r"$M$")
#ax.set_xlim([3e1,3e3])
ax.set_ylim([0.8e0, 2e2])
fig.set_size_inches(size * cm, size * cm)
plt.legend(frameon=False, loc='upper left')
plt.savefig("Performance/MED_data_time.pdf", format='pdf', bbox_inches="tight", dpi=300);









#plt.show()
