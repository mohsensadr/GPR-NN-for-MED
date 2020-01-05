import pickle
import numpy as np
from scipy import integrate
from random import randint
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.pyplot as plt

import math
import os.path

from numpy import ma

address = "ex_kl.txt"
ex = np.loadtxt(address,skiprows=0, unpack=True);

address = "GPR_kl.txt"
gpr = np.loadtxt(address,skiprows=0, unpack=True);


address = "NN_kl.txt"
NN = np.loadtxt(address,skiprows=0, unpack=True);

address = "GPR_err.txt"
gpr_err = np.loadtxt(address,skiprows=0, unpack=True);


address = "NN_err.txt"
NN_err = np.loadtxt(address,skiprows=0, unpack=True);


size = 8;
cm = 0.393701; #inches
plt.rcParams.update({'font.size': 9,'font.family': 'serif'})


fig, ax = plt.subplots();
ax.set_ylabel(r"$D_{KL}(f^{\mathrm{bi}}||f^\lambda_N)$")
ax.set_xlabel(r"$N$");

plt.plot(ex[0],ex[1],color="black", label ="ex", marker=">", linestyle="-.",markersize=7)
plt.plot(gpr[0],gpr[1],color="red", label ="GPR", marker="o",markersize=4)
plt.plot(NN[0],NN[1],color="blue", label ="NN", marker="s",markersize=4)
name = "KL_GPR_NN"
plt.legend( numpoints=1, loc='upper right' );
ax.set_yscale("log")
plt.xticks(gpr[0])
plt.ylim(1e-3,1e0)
plt.tight_layout();
fig.set_size_inches(0.9*size*cm, 0.9*size*cm)
plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
#plt.show();



fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel(r"$||\lambda^{\mathrm{ex}}-\lambda^{\mathrm{est}}||_2/||\lambda^{\mathrm{ex}}||_2$")
ax1.set_xlabel(r"$N$");
ax1.plot(gpr_err[0],gpr_err[1],color="red", label ="GPR", marker="o",markersize=4)
ax1.plot(NN_err[0],NN_err[1],color="blue", label ="NN", marker="s",markersize=4)
name = "Err_GPR_NN"
ax1.legend( numpoints=1, loc='lower left' );
ax1.set_ylim(1e-6,1e-1)
ax1.set_xticks(gpr_err[0])
#ax.set_yscale("log")
#plt.xticks(gpr[0])
#plt.ylim(1e-3,1e0)
#ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
ax2 = ax1.twinx()
ax2.plot(gpr_err[0],gpr_err[2],color="red", label ="GPR", marker="o", linestyle="-.",markersize=4)
ax2.plot(NN_err[0],NN_err[2],color="blue", label ="NN", marker="s", linestyle="-.",markersize=4)
ax2.set_ylabel(r"$||p^{\mathrm{ex}}-p^{\mathrm{est}}||_2$")
ax2.set_ylim(1e-6,1e-1)
#ax2.legend( numpoints=1, loc='upper right' );
plt.tight_layout();
ax1.set_yscale("log")
ax2.set_yscale("log")
fig.set_size_inches(size*cm, 0.85*size*cm)
plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
plt.show();

'''
fig, ax = plt.subplots();
ax.set_ylabel(r"$D_{KL}(f^{\mathrm{bi}}||f^\lambda_N)$")
ax.set_xlabel(r"$N$");
plt.plot(gpr_err[0],gpr_err[1],color="blue", label ="GPR", marker="o")
plt.plot(NN_err[0],NN_err[1],color="red", label ="NN", marker="s")

plt.legend( numpoints=1, loc='upper right' );
ax.set_yscale("log")
plt.xticks(gpr[0])
#plt.ylim(1e-3,1e0)
plt.tight_layout();
fig.set_size_inches(size*cm, size*cm)
plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
plt.show();
'''