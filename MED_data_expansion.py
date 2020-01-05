import numpy as np
import gpflow as gp
from scipy import integrate
#from integration_max_entropy import moments
#from sampling import samples
from random import randint
from numpy import linalg as LA
from numpy.linalg import inv
#from generating_samples_smart import sample_new_Nl
from changing_mean_Nl import sample_new
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import tensorflow as tf
import time
import os
#from scipy.optimize import minimize

#NN = [100,200 , 300, 400, 500, 600, 700, 800, 900, 1000]
NN=4000
dimy = [4]
val = 10
repeats = 1;
pmin = np.array([-1e-16,1.0+1e-16,-0.6,  0,-10, 0,-25,0]);
pmax = np.array([ 1e-16,1.0-1e-16, 0.6,1.5, 10,15, 25,110])

for j in range(len(dimy)):
    dimY = dimy[j]
    dimX = dimY - 2;

    name_file = "data_test/" + str(dimY) + "_"+str(val)+"_N_"+str(NN)+".txt"
    f = open(name_file, "a");
    st0 = "#"
    for i in range(0, 3 * dimY):
        if i == 0:
            st0 += "{:<13}".format("q" + str(i + 1))
        elif i < 2*dimY:
            st0 += "{:<14}".format("q" + str(i + 1))
        else:
            st0 += "{:<14}".format("lamb" + str(i - dimY + 1))
    st0 += "\n"
    if os.stat(name_file).st_size == 0:
        f.write(st0);


    for ii in range(NN):
        done = 1
        while (done == 1):
            La, Mo, La0 = sample_new(dimY, val)
            if np.all(pmin[2:dimY] < Mo[0][2:dimY]) and np.all(Mo[0][2:dimY] < pmax[2:dimY]):
                done = 0
            else:
                print("    REJECT!    ")
        Mo = np.array(Mo)
        La = np.array(La)
        st = "";
        mo = ['{:.16e}'.format(float(x)) for x in Mo[0, :]]
        la = ['{:.16e}'.format(float(x)) for x in La[0, :]]
        for j in range(0, 2*dimY):
            st += mo[j] + "  "
        for j in range(0, dimY):
            st += la[j] + "  "
        st += "\n"
        f.write(st);
    f.close()