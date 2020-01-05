import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate




size = 7;
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})
msize = 4
DIM = "1D"
if DIM == "3D":
    lab = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$", r"$c_{11}$", r"$c_{12}$", r"$c_{13}$", r"$c_{22}$", r"$c_{23}$",
           r"$c_{33}$",
           r"$\beta_{1}$", r"$\beta_{2}$", r"$\beta_{3}$", r"$\gamma$"]
    dimY = 13
elif DIM == "1D":
    dimY = 5;
    lab = []
    for i in range(dimY):
        lab.append("i="+str(i+1))
#methods = ["RBFArcCosine", "RBF_wom.txt", "Matern12", "Matern32", "Matern52", "Linear", "ArcCosine", "RBFLinear"]
methods = ["Matern12", "Matern32", "Matern52","RBF" ]
#methods = ["RBF_wom.txt"]
rel_total_error = [[],[],[],[],[],[],[],[],[]];
Var = [[],[],[],[],[],[],[],[],[]];
N = [[],[],[],[],[],[],[],[],[]];
#methods = ["RBF"]
#meth = "RBFArcCosine"
i=0

#import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True

for meth in methods:
    print(meth)

    if DIM == "3D":
        address = 'error/Relerror_post'+meth+'_3D.txt'
    else:
        address = 'error/Relerror_post' + meth + '.txt'
    x = np.loadtxt(address,skiprows=0, unpack=True);
    x = x[:,0:-1]
    for j in range(len(x[0])):
        rel_total_error[i].append(np.sqrt(np.sum(x[2+dimY:2+2*dimY,j]**2)))
        #Var[i].append(x[17,j])
        Var[i].append(np.sqrt(np.sum(x[2+2*dimY:2+3*dimY,j]**2)))
        N[i].append(x[0,j]);
    i=i+1

    if DIM=="3D":
        nc = 3
    else:
        nc = 1

    fig, ax = plt.subplots();
    [plt.plot(x[0],x[p+2],label=lab[p],marker="o", markersize=msize) for p in range(dimY)]
    ax.set_ylabel(r"$\mathrm{{E}}$")#[{|\lambda_p-\lambda^\mathrm{ex}_p|]}$")
    ax.set_xlabel("L")
    ax.set_yscale("log")
    ax.set_xscale("log")

    plt.legend(ncol=nc)
    name = "Fig"+DIM+"/Error_"+meth
    fig.set_size_inches(size*cm, size*cm)
    plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
    #plt.show()

    fig, ax = plt.subplots();
    [plt.plot(x[0],x[p+7],label=lab[p],marker="o", markersize=msize) for p in range(dimY)]
    ax.set_ylabel(r"$\mathrm{{E}}[{|\lambda_i-\lambda^\mathrm{ex}_i|/|\lambda^\mathrm{ex}_i|]}$")
    ax.set_xlabel("M")
    ax.set_yscale("log")
    ax.set_xscale("log")
    if meth == "RBF" and DIM=="1D":
        ax.set_ylim([7e-6, 0.2])
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.0)
    plt.legend(frameon=False,ncol=nc)
    name = "Fig"+DIM+"/RelError_"+meth
    fig.set_size_inches(size*cm, size*cm)
    plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
    #plt.show()

    fig, ax = plt.subplots();
    [plt.plot(x[0],x[p+12],label=lab[p],marker="o", markersize=msize) for p in range(dimY)]
    ax.set_ylabel(r"$\mathrm{Var}(|\lambda_i-\lambda^\mathrm{ex}_i|/|\lambda^\mathrm{ex}_i|)$")
    ax.set_xlabel("M")
    ax.set_yscale("log")
    ax.set_xscale("log")
    #ax.set_xlim([80,1100])
    if meth == "RBF" and DIM=="1D":
        ax.set_ylim([1e-5, 5e2])
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.0)
    plt.legend(frameon=False,ncol=nc)
    ax.tick_params(axis='y', which='minor')
    name = "Fig"+DIM+"/VarRelError_"+meth
    fig.set_size_inches(size*cm, size*cm)
    plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
    #plt.show()

    fig, ax = plt.subplots();
    #[plt.plot(x[0],x[p+17],label='p='+str(p+1),marker="o", markersize=7) for p in range(dimY)]
    plt.plot(x[0], x[17], marker="o", markersize=msize)
    ax.set_ylabel(r"$\mathrm{{E}}[{var(|\lambda_p-\lambda^\mathrm{ex}_p|)]}$")
    ax.set_xlabel("L")
    ax.set_yscale("log")
    ax.set_xscale("log")
    #ax.set_xlim([80,1100])
    #plt.legend()
    name = "Fig"+DIM+"/Var_"+meth
    fig.set_size_inches(size*cm, size*cm)
    plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
    
    #plt.show()
    #plt.close('all')

#plt.rcParams['mathtext.fontset'] = 'dejavuserif'
label_methods = [r"Mat$\mathrm{\'{e}}$rn$(12)$", r"Mat$\mathrm{\'{e}}$rn$(32)$", r"Mat$\mathrm{\'{e}}$rn$(52)$","RBF" ]

fig, ax = plt.subplots();
[plt.plot(N[idd],rel_total_error[idd],label=label_methods[idd],marker="o", markersize=3) for idd in range(len(methods))]
ax.set_ylabel(r"$||\mathrm{{E}}[{|\lambda-\lambda^\mathrm{ex}|/|\lambda^\mathrm{ex}|]}||_2$")
ax.set_xlabel("M")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylim([2e-5,0.5])
leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
plt.legend(frameon=False, prop={'family':'serif'})
name = "Fig1D/comparison"
fig.set_size_inches(size * cm, size * cm)
plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
#plt.show()



fig, ax = plt.subplots();
[plt.plot(N[idd],Var[idd],label=label_methods[idd],marker="o", markersize=3) for idd in range(len(methods))]
ax.set_ylabel(r"$||\mathrm{Var}({|\lambda-\lambda^\mathrm{ex}|/|\lambda^\mathrm{ex}|)}||_2$")
#ax.set_ylabel(r"$\mathrm{Var}(|\lambda_i^*-\lambda^\mathrm{*,ex}_i|)$")
ax.set_xlabel("M")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylim([5e-7,3e2])
leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
plt.legend(frameon=False)
name = "Fig1D/variance"
fig.set_size_inches(size * cm, size * cm)
plt.savefig(name + ".pdf", format='pdf', bbox_inches="tight", dpi=300);
#plt.show()
