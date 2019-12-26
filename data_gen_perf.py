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
#from scipy.optimize import minimize

NN = [100,200 , 300, 400, 500, 600, 700, 800, 900, 1000]
dimy = [4]
repeats = 1;

for j in range(len(dimy)):
    dimY = dimy[j]
    dimX = dimY - 2;

    fp = open("data_geb_perf_dimY_" + str(dimY)+".txt", "a")
    st = ""

    #if dimY == 4:
    #    la_min = np.array([-0.1,-0.1,-0.1,1e-4])
    #    la_max = np.array([0.1 , 0.1, 0.1,1e-4])
    #la_min = -np.ones(dimY) * 0.01;
    #la_max =  np.ones(dimY) * 0.01;

    k=0
    start_time = time.time()
    dt = [];
    [dt.append([]) for _ in range(len(NN))];
    for ii in range(NN[-1]):
        La, Mo, La0 = sample_new(dimY)
        if ii == NN[k]-1:
            end_time = time.time()
            dt[k] = end_time-start_time

            print("dimY="+str(dimY)+", N1="+str(ii+1)+",  dt="+str(dt[k]))
            st += str(dimY) + "  " + str(ii+1) + "  "+str(dt[k])+"\n"
            k = k + 1
    fp.write(st);
    fp.close()