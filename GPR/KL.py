import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from numpy import ma
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

msize = 0
size = 6;
lw = 1
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})



markers = ["<","s","o"]
mu = [0.8,0.9,0.95]
sig =  [0.3, 0.2, 0.15]
fig, ax = plt.subplots();
fig2, ax2 = plt.subplots();
fig3, ax3 = plt.subplots();
test  = ["(a)","(b)", "(c)"]
for j in range(len(mu)):
    mu1 = mu[j]
    sig1 = sig[j]
    case = "kl_mu1_"+str(mu1)
    x = np.loadtxt(case, skiprows=0, unpack=True);
    l = x[0];
    kl = x[1];
    var = x[2];
    erp = x[3]
    ax.plot(l, kl, label=test[j], marker=markers[j],
             markersize=3)
    ax2.plot(l, var, label=test[j], marker=markers[j],
            markersize=3)
    ax3.plot(l, erp, label=test[j], marker=markers[j],
             markersize=3)


ax.set_ylabel(r"$D_{KL}(f^{\mathrm{bi}}_\epsilon||f^\lambda_N)$")
ax.set_yscale("log")
ax.set_xlabel(r"$N$")
name = "TestCase1_with_noise/KL"
ax.set_ylim([1e-3, 0.4])
plt.xticks(l)
fig.set_size_inches(size * cm, size * cm)
ax.legend(frameon=False,loc="lower left")
fig.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);

ax2.set_ylabel(r"$||\mathrm{Var}(\lambda^{\mathrm{est}})||_2$")
ax2.set_yscale("log")
ax2.set_xlabel(r"$N$")
name = "TestCase1_with_noise/var"
#ax.set_ylim([1e-3, 0.4])
plt.xticks(l)
fig2.set_size_inches(size * cm, size * cm)
ax2.legend(frameon=False,loc="upper left")
fig2.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);

ax3.set_ylabel(r"$||p^{\mathrm{in}}-p^{\mathrm{est}}||_2/||p^{\mathrm{in}}||_2$")
ax3.set_yscale("log")
ax3.set_xlabel(r"$N$")
name = "TestCase1_with_noise/p"
#ax.set_ylim([1e-3, 0.4])
plt.xticks(l)
fig3.set_size_inches(size * cm, size * cm)
ax3.legend(frameon=False,loc="lower right")
fig3.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);


size = 6;
markers = ["<","s","o"]
dimy = [4,6,8]
fig, ax = plt.subplots();
for j in range(len(dimy)):
    dimY = dimy[j]
    case = "Boltz_dimY_"+str(dimY)
    x = np.loadtxt(case, skiprows=0, unpack=True);
    t = np.array(x[0]);
    kl = np.array(x[1]);
    plt.plot(t, kl, label=r"$N="+str(dimY)+"$", marker=markers[j],
             markersize=3)

plt.legend(frameon=False,loc="lower left")
ax.set_ylabel(r"$D_{KL}(f^{\mathrm{Bolt}}||f^\lambda_N)$")
ax.set_yscale("log")
#ax.set_xscale("log")
ax.set_xlabel(r"$\hat{t}$")
name = "TestCase_Boltz/KL_Boltz"
#ax.set_ylim([1e-3, 1.0])
#plt.xticks(t)
fig.set_size_inches(size * cm, size * cm)
plt.legend(frameon=False, loc='lower left')
plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);


colors = ['r', 'b', 'g', 'm', 'c']
y = np.loadtxt("Boltz_ex_mom.txt", skiprows=0, unpack=True);
z = y[:,-1]
dimy = [4,6,8]
for j in range(len(dimy)):
    fig, ax = plt.subplots();
    dimY = dimy[j]
    case = "Boltz_dimY_"+str(dimY)
    x = np.loadtxt(case, skiprows=0, unpack=True);

    t = np.array(x[0]);
    k = 0;
    for i in range(dimY):
        if i%2==1:
            plt.plot(t, abs(x[2+i]/z[1+i]), label=r"$\hat{p}_"+str(i+1)+"^{("+str(dimY)+")}$", marker=markers[j],
                 markersize=3,color=colors[k], linewidth=0.5)
            plt.plot(t, abs(y[1 + i] / z[1 + i]), label='_nolegend_',
                     marker=markers[j],markersize=3, linestyle="--",color=colors[k], linewidth=1.0)
            k = k+1

    ax.set_ylabel(r"$\hat{p}_i(\hat{t})/\hat{p}_i(\hat{t}_\mathrm{final})$")
    #ax.set_yscale("log")
    #ax.set_xscale("log")
    ax.set_xlabel(r"$\hat{t}$")
    name = "TestCase_Boltz/mom_Boltz"+str(dimY)
    #ax.set_ylim([1e-3, 1.0])
    #plt.xticks(t)
    fig.set_size_inches(size * cm, size * cm)
    plt.legend(frameon=False, loc='lower right')
    plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
#plt.show();

