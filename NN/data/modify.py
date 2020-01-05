import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
from scipy import integrate

def H1(v):
    return v
def H2(v):
    return v**2
def H3(v):
    return v**3
def H4(v):
    return v**4
def H5(v):
    return v**5
def H6(v):
    return v**6
def zhi(v,l,i):
    return Z(v,l)*v**i
def Z(v, l):
    return np.exp(-H1(v) * l[0] - H2(v) * l[1] - H3(v) * l[2] - H4(v) * l[3] - H5(v) * l[4] - H6(v) * l[5])

address = "8l1e-7_var8th.txt"
dimX = 8; dimY = 8;
xx = np.loadtxt(address,skiprows=1, unpack=True);
N = len(xx[0])-1
i0=0; i1=20; i2=26

name_file = "8l.txt"
f = open(name_file, "w");
st0="#"
for i in range(0,i2):
    if i==0:
        st0 += "{:<13}".format("q" + str(i + 1))
    elif i<i1:
        st0 += "{:<14}".format("q" + str(i + 1))
    else:
        st0+="{:<14}".format("lamb"+str(i-i1+1))
st0+= "\n"
if os.stat(name_file).st_size ==0:
    f.write(st0);
for i in range(0,N):
    st = "";
    mo = ['{:.16e}'.format(float(x)) for x in xx[i0:i0+dimX,i]]
    la = ['{:.16e}'.format(float(x)) for x in xx[i1:i1+dimY, i]]
    for j in range(0,dimX):
        st += mo[j] + " "
    for j in range(0,dimY):
        st += la[j] + " "
    st += "\n"
    f.write(st);

print("look!")

print('done!');


