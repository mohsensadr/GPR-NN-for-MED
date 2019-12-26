import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

msize = 0
size = 5;
lw = 1
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})



markers = ["<","s","o"]
mu = [0.8,0.9,0.95]
sig =  [0.3, 0.2, 0.15]
fig, ax = plt.subplots();
test  = ["(a)","(b)", "(c)"]
for j in range(len(mu)):
    mu1 = mu[j]
    sig1 = sig[j]
    case = "noise_kl_mu1_"+str(mu1)
    x = np.loadtxt(case, skiprows=0, unpack=True);
    l = x[0];
    kl = x[1];
    plt.plot(l, kl, label=test[j], marker=markers[j],
             markersize=3)


ax.set_ylabel(r"$D_{KL}(f^{\mathrm{bi}}_\epsilon||f^\lambda_N)$")
ax.set_yscale("log")
ax.set_xlabel(r"$N$")
name = "TestCase1_with_noise/KL"
ax.set_ylim([1e-3, 0.4])
plt.xticks(l)
fig.set_size_inches(size * cm, size * cm)
plt.legend(frameon=False,loc="lower left")
plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);


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
ax.set_ylabel(r"$D_{KL}(f^{\mathrm{Bolt.}}||f^\lambda_N)$")
ax.set_yscale("log")
#ax.set_xscale("log")
ax.set_xlabel(r"$t$")
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

    ax.set_ylabel(r"$\hat{p}(t)$")
    #ax.set_yscale("log")
    #ax.set_xscale("log")
    ax.set_xlabel(r"$t$")
    name = "TestCase_Boltz/mom_Boltz"+str(dimY)
    #ax.set_ylim([1e-3, 1.0])
    #plt.xticks(t)
    fig.set_size_inches(size * cm, size * cm)
    plt.legend(frameon=False, loc='lower right')
    plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
#plt.show();

