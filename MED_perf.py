import numpy as np
import gpflow as gp
from scipy import integrate
#from integration_max_entropy import moments
#from sampling import samples
from random import randint
from numpy import linalg as LA
from numpy.linalg import inv

import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import tensorflow as tf
import time
#from scipy.optimize import minimize

def Remove_Bad_Data(xx):
    xx2 = []
    xx_bad = []
    for i in range(0,len(xx[0])):
        sc = 10000000
        if abs(xx[20, i]-1) < sc and abs(xx[21, i]-1) < sc and abs(xx[22, i]-1) < sc and abs(xx[23, i]-1) < sc:
            xx2.append(xx[:,i])
        elif abs(xx[20, i]-1) < 2 and abs(xx[21, i]-1) < 2 and abs(xx[22, i]-1) < 2 and abs(xx[23, i]-1) < 2 :
            xx_bad.append(xx[:, i])
    return xx2, xx_bad
def transf_to(xx,m,v,n):
    for i in range(2,n):
        if abs(v[i])>1e-15:
            xx[i,:] = (xx[i,:] - m[i] ) / v[i] ** 0.5+1.0
    return xx

def transf_back(xx,m,v,n):
    for i in range(2,n):
        if abs(v[i]) > 1e-15:
            xx[i,:] = xx[i,:]* v[i] ** 0.5 + m[i]
    return xx

NN = [125, 250, 500, 1000, 2000]
dimy = [8]
repeats = 5;
for j in range(len(dimy)):
    dimY = dimy[j]
    dimX = dimY - 2;
    i0 = 2;
    i1 = i0 + dimX;
    i11 = 20;
    i2 = i11 + dimY;

    fp = open("MED_perf_dimY_" + str(dimY)+".txt", "a")
    st = ""
    '''
            ----    Reading Data      ----
    '''
    address = "data/" + str(dimY) + "l.txt"
    x0 = np.loadtxt(address, skiprows=1, unpack=True);
    dim = len(x0[:, 0]);  ## 26 here

    vvar = []
    mm = []
    for i in range(dim):
        mm.append(np.mean(x0[i, :]));
        vvar.append(np.var(x0[i, :]));
    x0 = transf_to(x0, mm, vvar, dim);
    x, x_bad = Remove_Bad_Data(x0)
    x = np.array(x)
    x_bad = np.array(x_bad)

    for k in range(len(NN)):
        N1 = NN[k]
        ytrain = [];
        xtrain = [];
        for i in range(N1):
            xtrain.append(x[i, i0:i1].copy())
            ytrain.append(x[i, i11:i2].copy())
        print("Training data is set for "+str(N1))

        dt = [];
        [dt.append([]) for _ in range(repeats)];
        for repeat in range(repeats):
            y2 = np.array(ytrain)
            len_sc = []
            for i in range(0, dimX):
                len_sc.append(i);

            k1 = gp.kernels.RBF(input_dim=dimX, active_dims=len_sc, ARD=True)
            kernel = k1
            model = gp.models.GPR(xtrain, ytrain, kernel)

            model.likelihood.variance = 1e-10
            model.likelihood.variance.trainable = False


            start_time = time.time()

            opt = gp.train.ScipyOptimizer(tol=1e-12, options={"eps": 1E-12, "disp": True, 'ftol': 1e-12})
            # opt = gp.train.ScipyOptimizer(method='SLSQP',tol=1e-12, options={"eps": 1E-12})
            # opt = gp.train.ScipyOptimizer(method='SLSQP',options={'maxiter': 5000, 'ftol': 1e-16, 'disp': True, 'eps': 1e-16})
            opt.minimize(model, disp=True, maxiter=10000)

            end_time = time.time()
            dt[repeat] = end_time-start_time

        print("dimY="+str(dimY)+", N1="+str(N1)+",  dt="+str(np.median(dt)))
        st += str(dimY) + "  " + str(N1) + "  "+str(np.median(dt))+"\n"
    fp.write(st);
    fp.close()
