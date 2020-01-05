
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
# import matplotlib.pyplot as plt
import os
import random
import shutil
import time
from scipy import integrate

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
    print("D="+str(D))
    return q;
####################################
# Test the trained neural network
####################################

# Get the overall best model
dimX = 6;
dimx = "6"

mad = 'functions_'+dimx
model = keras.models.load_model(mad+'/my_model.h5')

## Load the feature scaling related arrays

mean_input=np.genfromtxt(mad+'/mean_input.dat', dtype=np.float64)
deviation_input=np.genfromtxt(mad+'/deviation_input.dat', dtype=np.float64)
mean_output=np.genfromtxt(mad+'/mean_output.dat', dtype=np.float64)
deviation_output=np.genfromtxt(mad+'/deviation_output.dat', dtype=np.float64)

#x_predict=np.array([[-2.0776612794713592e-01,2.4657657296865780e+00],[-8.8954306249590817e-02,2.7204257513325385e+00]])
#y_predict=np.array([[-1.9869638495414793e-01,2.2533577982000172e-01,8.3116051286787548e-02,6.0948059010733829e-02],
#    [-6.1401350751308620e-02,4.0001633391736557e-01,2.2712736497634407e-02,1.8933481204079984e-02]])

#q = np.array([-1.1102230246251565e-16, 1.0000000000000002, -0.26999999999999985, 1.7177999999999987, -1.2419999999999987, 4.270055999999998, -5.266673999999992, 14.256776279999983, -24.04980719999996, 60.05188856159989])
if dimX == 3:
    #x_predict = np.array([[1.3136062500214005e+00]])
    #y_predict = np.array([[1.6190676471471449e-01,  5.8465463351464497e-01,  -5.5633305061260560e-02]])
    x_predict = np.array([[-0.26999999999999985]])
    y_predict = np.array([[-0.1236756900314129,  0.516679580252206, 0.04122636998331299]])
elif dimX == 4:
    x_predict = np.array([[-0.26999999999999985, 1.7177999999999987]])
    y_predict = np.array([[-0.7043640854047318, -1.213225498156559, 0.432670353803256, 0.5496731259305444]])
elif dimX == 6:
    x_predict = np.array([[-0.26999999999999985, 1.7177999999999987, -1.2419999999999987, 4.270055999999998]])
    y_predict = np.array([[-0.04743950102790144,-1.7963760730592575,-0.5575983379196991,0.7722133489203675,0.32662138992668466,0.03369916619152445]])
elif dimX == 8:
    x_predict = np.array([[1.2553724307749804e-11, 1.0000000000865887, -0.27000000000345903, 1.7177999997325288, -1.241999999972635, 4.270056000537639, -5.266674000017453, 14.25677627953217]])
    y_predict = np.array([[-0.21478409567528478, -3.0224417983240226, -0.24331151837042694,  2.0160747438695665, 0.26935442896527784, -0.335545383359783, -0.029866218468154883,  0.018967782958135355]])
#x_predict = np.array([[0.018367]])
#y_predict = np.array([[-0.030528393102483158, 0.51074697618138998, 0.0099454107351601897]])
#mean_input = np.array(mean_input)
#deviation_input = np.array(deviation_input)

## Feature Scaling
for j in range(len(x_predict[0,:])):
    if dimX !=3:
        x_predict[:,j][0]= (x_predict[:,j][0] - mean_input[j]) / deviation_input[j]
    else:
        x_predict[:, j][0] = (x_predict[:, j][0] - mean_input) / deviation_input


test_loss, test_mse = model.evaluate(x_predict,y_predict)
print('Prediction accuracy:', test_mse)

print(y_predict)

output = model.predict(x_predict)

## Output scaling back
for j in range(len(output[0,:])):
    output[:,j]= output[:,j]*deviation_output[j] + mean_output[j]

print(output)
print("rel error = "+str((y_predict-output)/y_predict))
'''
W = model.get_weights();


q = Mom(output[0], dimX, dimX);
print("mom= "+str(q))
Npredict=len(output[:,0])

relative_error= 0.0
enorm_sum=0.0
error_sum=0.0
mse=0.0
for i in range(Npredict):
    enorm= np.linalg.norm(y_predict[i,:])
    error= np.linalg.norm(output[i,:] - y_predict[i,:])
    rerror= error/enorm
    relative_error += rerror

    mse += error**2

    enorm_sum += enorm**2
    error_sum += error**2

print('mse=',mse/Npredict)

relative_error /= Npredict
print('Relative error of the predicted memory term is:', relative_error)

relative_error2= np.sqrt(error_sum/enorm_sum)
print('Relative error of the predicted memory term is:', relative_error2)

sys.exit()

'''