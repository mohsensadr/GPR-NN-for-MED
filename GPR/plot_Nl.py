import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def Z(v,l,dimY):
    f = 0.0;
    for i in range(1,dimY+1):
        f += l[i-1]*v**i
    return np.exp(-f);

def HiZ(v,l,dimY,i):
    return v**i*Z(v,l,dimY)

from numpy import ma
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

msize = 0
size = 7;
lw = 1
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})


dimy = [4,6,8]
import seaborn as sns

colors = ['darkred', 'darkblue', 'darkgreen']
lcolors = ['lightsalmon', 'lightblue', 'greenyellow']

val = 10
for j in range(2,8):
    fig, ax = plt.subplots()
    for i in reversed(range(len(dimy))):
        dimY = dimy[i];
        address = "expansion/"+str(dimY)+"_"+str(val)+".txt"
        #if dimY==8:
        #    address = "expansion/" + str(dimY) + "_mod3.txt"
        x = np.loadtxt(address,skiprows=1, unpack=True);
        #ax.plot(abs(x[j+2]), '.', label=r"$p_"+str(j+3)+"^"+str(dimY)+"$")
        mom = j+1
        if mom==3:
            Lmin=-5.0; Lmax=5.0;
        elif mom==4:
            Lmin = 0;
            Lmax = 40.0;
            ax.set_yscale("log")
            plt.ylim(1e-4, 1e0)
        elif mom==5:
            Lmin = -75;
            Lmax = 75.0;
        elif mom==6:
            Lmin = 0;
            Lmax = 200.0;
            ax.set_yscale("log")
        elif mom==7:
            Lmin = 0;
            Lmax = 1e3;
            ax.set_yscale("log")
        elif mom==8:
            Lmin = 0;
            Lmax = 1e3;
            ax.set_yscale("log")
        else:
            Lmin = -200;
            Lmax = 200;
        plt.xlim(Lmin, Lmax)
        ax.hist(x[j], label=r"$N=" + str(dimY) + "$", density=True, alpha=0.3, histtype='stepfilled',
                bins=np.arange(Lmin, Lmax + (Lmax-Lmin)/40.0, (Lmax-Lmin)/40.0),color=lcolors[i], edgecolor = colors[i])
        #sns.distplot(x[j], label=r"$p_" + str(mom) + "^" + str(dimY) + "$",bins=np.arange(Lmin, Lmax + (Lmax-Lmin)/50.0, (Lmax-Lmin)/50.0))
        #ax.set_yscale("log")
    plt.legend(frameon=False)
    #plt.text(0.15, 0.9, r"$b="+str(val)+"$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_ylabel(r"$f(p_"+str(mom)+"|b="+str(val)+")$")
    ax.set_xlabel(r"$p_" + str(mom) + "$")
    fig.set_size_inches(size * cm, size * cm)
    name = "expansion/p_"+str(mom)+"_"+str(val)
    plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);

#ax.set_ylabel(r"$\frac{\exp(-\sum_{i=1}^"+dimy+" \lambda_i v^i)}{\int_{-10}^{10} \exp({-\sum_{i=1}^"+dimy+" \lambda_i v^i}) dv}$")
#plt.xlim(-10.0,10.0)
#plt.ylim(-0.1,5.0)
#name = "distrs_inverse"
#fig.set_size_inches(size*cm, size*cm)
#plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
#plt.show()

