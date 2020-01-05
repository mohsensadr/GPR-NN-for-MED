
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

# set random seed
def seed_random_number(seed):
    # see https://stackoverflow.com/a/52897216
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)

random_seed = 1
seed_random_number(random_seed)

##################################
## Work mode: 0-Test, 1-Train
##################################

NN_model= 1 # 0: test; 1: train

###########################
## Data
###########################

f = open('parameters.in', 'r')
num_input= int(f.readline())
num_output= int(f.readline())
num_exclude= int(f.readline())
f.close() 

if(num_input > num_output):
    print('Are you sure you want more moments than lambdas?') 
    sys.exit()
elif(num_input < num_output):
    print('Error: number of moments is less than number of lambdas!') 
    sys.exit()

f=open('network_info.in', 'r')
r0= int(f.readline())
m0= int(f.readline())
number_of_trains= int(f.readline())
num_restarts= int(f.readline())
training_epochs= int(f.readline())
batch_size= int(f.readline())
learning_rate= float(f.readline())
lambda_l2= float(f.readline())
f.close()

# training_data_file   = 'training.dat'
# validation_data_file = 'validation.dat'
# prediction_data_file = 'prediction.dat'
# data_file   = '*'+str(num_output)+'th.txt'
suf = ""
data_file= 'data/8l'+suf+'.txt'

Data       = np.genfromtxt(data_file, dtype=np.float64)  
Ndata      = len(Data[:,0])
print('Entire data size:', Ndata)

print(Data[:2,:])
print(Data[-2:,:])

# sys.exit()

idx= np.array(range(Ndata))
# np.random.shuffle(idx)

Ntrain= int(0.8*Ndata)
Nvalidate= int(0.1*Ndata)
Npredict= Ndata - Ntrain - Nvalidate

Train= Data[idx[:Ntrain],:]
Validate= Data[idx[Ntrain:Ntrain+Nvalidate],:]
Predict= Data[idx[Ntrain+Nvalidate:],:]

 
print('Training data size:', Ntrain)
x_train= Train[:,num_exclude:num_input]
y_train= Train[:,-num_output:]

Model_save_path= 'functions_'+str(num_output)+suf

if not os.path.isdir(Model_save_path):
    os.makedirs(Model_save_path)

print('Validation data size:', Nvalidate)  
x_validate= Validate[:,num_exclude:num_input]
y_validate= Validate[:,-num_output:]

print('Test data size:', Npredict)
x_predict= Predict[:,num_exclude:num_input]
y_predict= Predict[:,-num_output:]

# print(y_predict[:2,:])

# we remove the first moment that is always zero
num_input -= num_exclude

##Feature scaling
mean_input= np.zeros(num_input, dtype=np.float64)
deviation_input= np.zeros(num_input, dtype=np.float64)

for i in range(Ntrain):
    for j in range(num_input):
        mean_input[j]+=x_train[i,j]

mean_input/= Ntrain

# print(x_train[:10,:])
# sys.exit()

for i in range(Ntrain):
    for j in range(num_input):
        deviation_input[j]+= (x_train[i,j]-mean_input[j])**2

deviation_input/=Ntrain
deviation_input=np.sqrt(deviation_input)

print('Mean of input variables:',mean_input)
print('Deviation of input variables:',deviation_input)

## Feature Scaling
for j in range(num_input):

    x_train[:,j]= (x_train[:,j] - mean_input[j]) / deviation_input[j]
    x_validate[:,j]= (x_validate[:,j] - mean_input[j]) / deviation_input[j] 
    x_predict[:,j]= (x_predict[:,j] - mean_input[j]) / deviation_input[j]


## Feature scaling of output
mean_output= np.zeros(num_output, dtype=np.float64)
deviation_output= np.zeros(num_output, dtype=np.float64)

scale_output=0.0
for i in range(Ntrain):
    scale_output+= np.linalg.norm(y_train[i,:])

scale_output /= Ntrain*np.sqrt(1.0*num_output)
print('output scale:',scale_output)

deviation_output[:]= scale_output

mean_output[:]=0.0

print('Mean of output variables:',mean_output)
print('Deviation of output variables:',deviation_output)

## Feature Scaling
for j in range(num_output):

    y_train[:,j]= (y_train[:,j] - mean_output[j]) / deviation_output[j]
    y_validate[:,j]= (y_validate[:,j] - mean_output[j]) / deviation_output[j] 
    y_predict[:,j]= (y_predict[:,j] - mean_output[j]) / deviation_output[j]


np.savetxt(Model_save_path+'/mean_input.dat',mean_input,delimiter=',')
np.savetxt(Model_save_path+'/deviation_input.dat',deviation_input,delimiter=',')

np.savetxt(Model_save_path+'/mean_output.dat',mean_output,delimiter=',')
np.savetxt(Model_save_path+'/deviation_output.dat',deviation_output,delimiter=',')

# Training Parameters
if(NN_model==0):
    number_of_trains = 0

# network parameters
n1=10
n2=20
n3=40
# n4=10
# n5=10

def create_model():

    x = keras.layers.Input(shape=(num_input,))

    y1= keras.layers.Dense(n1, activation='tanh',  
        kernel_initializer='glorot_uniform', bias_initializer='RandomNormal', 
        kernel_regularizer=keras.regularizers.l2(lambda_l2))(x)

    y2= keras.layers.Dense(n2, activation='tanh',  
        kernel_initializer='glorot_uniform', bias_initializer='RandomNormal', 
        kernel_regularizer=keras.regularizers.l2(lambda_l2))(y1)

    y3= keras.layers.Dense(n3, activation='tanh',  
        kernel_initializer='glorot_uniform', bias_initializer='RandomNormal', 
        kernel_regularizer=keras.regularizers.l2(lambda_l2))(y2)

    # y4= keras.layers.Dense(n4, activation='tanh',  
    #     kernel_initializer='glorot_uniform', bias_initializer='RandomNormal', 
    #     kernel_regularizer=keras.regularizers.l2(lambda_l2))(y3)

    # y5= keras.layers.Dense(n5, activation='tanh',  
    #     kernel_initializer='glorot_uniform', bias_initializer='RandomNormal', 
    #     kernel_regularizer=keras.regularizers.l2(lambda_l2))(y4)

    y = keras.layers.Dense(num_output, activation='linear',
        kernel_initializer='glorot_uniform', bias_initializer='RandomNormal', 
        kernel_regularizer=keras.regularizers.l2(lambda_l2))(y3)

    model = keras.models.Model(inputs=x, outputs=y)

    adam=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, 
        decay=0.0, amsgrad=False)

    model.compile(optimizer=adam, loss='mse', metrics=['mse'])

    
    return model

model = create_model()
print(model.summary())

# keras.utils.plot_model(model, to_file='model.png')


best_accuracy= -1.0e10

if(r0>0 or m0>0):

    # Restore variables from disk
    model = keras.models.load_model(Model_save_path+'/my_model.h5')

    test_loss, test_mse = model.evaluate(x_predict,y_predict)
    print('Prediction accuracy:', test_mse)

    output= model.predict(x_predict)

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

    valid_loss, valid_mse = model.evaluate(x_validate, y_validate)
    print('Validation accuracy:', valid_mse)

    best_accuracy= -valid_mse


## Training
for r in range(number_of_trains):

    ## Creat the folder to save the model
    Model_dir= Model_save_path +'/model_'+str(r+1+r0) # be careful of the folder number
    if not os.path.isdir(Model_dir):
        os.makedirs(Model_dir) 

    max_val_acc = -1.0e10
    if(m0>0):
        model = keras.models.load_model(Model_dir+'/my_model.h5')
        valid_loss, valid_mse = model.evaluate(x_validate, y_validate)
        print('Validation accuracy:', valid_mse)
        max_val_acc= -valid_mse

    val_acc_list   = np.zeros(num_restarts)
    train_acc_list = np.zeros(num_restarts)

    for m in range(num_restarts):

        restart_dir= Model_dir+'/restart_'+str(m+1+m0) # be careful of the folder number
        if not os.path.isdir(restart_dir):
            os.makedirs(restart_dir) 

        model= create_model()

        model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=training_epochs, verbose=1, callbacks=None, 
            validation_split=0.0, validation_data=(x_validate,y_validate), 
            shuffle=True, class_weight=None, 
            sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)


        ## Save weights
        id_file=0
        for weight in model.get_weights(): # weights from Dense layer omitted
            id_file +=1

            if(id_file==1):
                np.savetxt(restart_dir+'/W1.txt',weight,delimiter=',')

            elif(id_file==2):
                np.savetxt(restart_dir+'/b1.txt',weight,delimiter=',')

            elif(id_file==3):
                np.savetxt(restart_dir+'/W2.txt',weight,delimiter=',')

            elif(id_file==4):
                np.savetxt(restart_dir+'/b2.txt',weight,delimiter=',')

            elif(id_file==5):
                np.savetxt(restart_dir+'/W3.txt',weight,delimiter=',')

            elif(id_file==6):
                np.savetxt(restart_dir+'/b3.txt',weight,delimiter=',')

            # elif(id_file==7):
            #     np.savetxt(restart_dir+'/W4.txt',weight,delimiter=',')

            # elif(id_file==8):
            #     np.savetxt(restart_dir+'/b4.txt',weight,delimiter=',')

            # elif(id_file==9):
            #     np.savetxt(restart_dir+'/W5.txt',weight,delimiter=',')

            # elif(id_file==10):
            #     np.savetxt(restart_dir+'/b5.txt',weight,delimiter=',')

            elif(id_file==7):
                np.savetxt(restart_dir+'/Wo.txt',weight,delimiter=',')

            elif(id_file==8):
                np.savetxt(restart_dir+'/bo.txt',weight,delimiter=',')

        ## copy files
        shutil.copy2('parameters.in', restart_dir)
        shutil.copy2(Model_save_path+'/mean_input.dat', restart_dir)
        shutil.copy2(Model_save_path+'/mean_output.dat', restart_dir)
        shutil.copy2(Model_save_path+'/deviation_input.dat', restart_dir)
        shutil.copy2(Model_save_path+'/deviation_output.dat', restart_dir)
        #shutil.copy2(Model_save_path+'/range_input.dat', restart_dir)

            
        # Performance of trained model
        train_loss, train_mse = model.evaluate(x_train, y_train)
        print('Training accuracy:', train_mse)

        valid_loss, valid_mse = model.evaluate(x_validate, y_validate)
        print('Validation accuracy:', valid_mse)

        test_loss, test_mse = model.evaluate(x_predict,y_predict)
        print('Prediction accuracy:', test_mse)

        output= model.predict(x_predict)

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


        val_acc_list[m]= -valid_mse

        if(val_acc_list[m] > max_val_acc):
            print("\n\t\t\tA better model was found with validation accuracy increasing from %.10f to %.10f "
                  %(max_val_acc,val_acc_list[m]))    

            max_val_acc = val_acc_list[m]

            ## save the overall best model 
            if(val_acc_list[m]>best_accuracy):
                model.save(Model_save_path+'/my_model.h5')  # creates a HDF5 file 'my_model.h5'
                best_accuracy= val_acc_list[m]

            ## Save model & weights in folder: model_r
            model.save(Model_dir+'/my_model.h5')
            id_file=0
            for weight in model.get_weights(): # weights from Dense layer omitted
                id_file +=1

                if(id_file==1):
                    np.savetxt(Model_dir+'/W1.txt',weight,delimiter=',')

                elif(id_file==2):
                    np.savetxt(Model_dir+'/b1.txt',weight,delimiter=',')

                elif(id_file==3):
                    np.savetxt(Model_dir+'/W2.txt',weight,delimiter=',')

                elif(id_file==4):
                    np.savetxt(Model_dir+'/b2.txt',weight,delimiter=',')

                elif(id_file==5):
                    np.savetxt(Model_dir+'/W3.txt',weight,delimiter=',')

                elif(id_file==6):
                    np.savetxt(Model_dir+'/b3.txt',weight,delimiter=',')

                # elif(id_file==7):
                #     np.savetxt(Model_dir+'/W4.txt',weight,delimiter=',')

                # elif(id_file==8):
                #     np.savetxt(Model_dir+'/b4.txt',weight,delimiter=',')

                # elif(id_file==9):
                #     np.savetxt(Model_dir+'/W5.txt',weight,delimiter=',')

                # elif(id_file==10):
                #     np.savetxt(Model_dir+'/b5.txt',weight,delimiter=',')

                elif(id_file==7):
                    np.savetxt(Model_dir+'/Wo.txt',weight,delimiter=',')

                elif(id_file==8):
                    np.savetxt(Model_dir+'/bo.txt',weight,delimiter=',')

        else:
            print("\n\t\t\tA better model was not found.")    

        print('Training:',r+1,'Restart:',m+1,'finihsed') 

    ## copy files
    shutil.copy2('parameters.in', Model_dir)
    shutil.copy2(Model_save_path+'/mean_input.dat', Model_dir)
    shutil.copy2(Model_save_path+'/mean_output.dat', Model_dir)
    shutil.copy2(Model_save_path+'/deviation_input.dat', Model_dir)
    shutil.copy2(Model_save_path+'/deviation_output.dat', Model_dir)
    #shutil.copy2(Model_save_path+'/range_input.dat', Model_dir)

    print('Finish training:',r+1)
    

######################
# Test model
######################

# Get the overall best model
model = keras.models.load_model(Model_save_path+'/my_model.h5')

test_loss, test_mse = model.evaluate(x_predict,y_predict)
print('Prediction accuracy:', test_mse)

print(y_predict[:2,:])

output = model.predict(x_predict)
# np.savetxt('output.txt',output_lstm,delimiter=',')

for j in range(num_output):
    output[:,j]= output[:,j]*deviation_output[j] + mean_output[j]
    y_predict[:,j]= y_predict[:,j]*deviation_output[j] + mean_output[j]

print(y_predict[:2,:])

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


## output the error 
f=open(Model_save_path+'error.dat','a+')
f.write(str(num_output)+', '+str(relative_error)+', ' + str(relative_error2) +'\n')
f.close()


## Save weights
id_file=0
for weight in model.get_weights(): # weights from Dense layer omitted
    id_file +=1
    # # print(len(weight.shape))
    # print(weight.shape)
    # print(weight)

    if(id_file==1):
        np.savetxt(Model_save_path+'/W1.txt',weight,delimiter=',')

    elif(id_file==2):
        np.savetxt(Model_save_path+'/b1.txt',weight,delimiter=',')

    elif(id_file==3):
        np.savetxt(Model_save_path+'/W2.txt',weight,delimiter=',')

    elif(id_file==4):
        np.savetxt(Model_save_path+'/b2.txt',weight,delimiter=',')

    elif(id_file==5):
        np.savetxt(Model_save_path+'/W3.txt',weight,delimiter=',')

    elif(id_file==6):
        np.savetxt(Model_save_path+'/b3.txt',weight,delimiter=',')

    # elif(id_file==7):
    #     np.savetxt(Model_save_path+'/W4.txt',weight,delimiter=',')

    # elif(id_file==8):
    #     np.savetxt(Model_save_path+'/b4.txt',weight,delimiter=',')

    # elif(id_file==9):
    #     np.savetxt(Model_save_path+'/W5.txt',weight,delimiter=',')

    # elif(id_file==10):
    #     np.savetxt(Model_save_path+'/b5.txt',weight,delimiter=',')

    elif(id_file==7):
        np.savetxt(Model_save_path+'/Wo.txt',weight,delimiter=',')

    elif(id_file==8):
        np.savetxt(Model_save_path+'/bo.txt',weight,delimiter=',')


## copy files
shutil.copy2('parameters.in', Model_save_path)
#shutil.copy2(Model_save_path+'/mean_input.dat', Model_save_path)
#shutil.copy2(Model_save_path+'/mean_output.dat', Model_save_path)
#shutil.copy2(Model_save_path+'/deviation_input.dat', Model_save_path)
#shutil.copy2(Model_save_path+'/deviation_output.dat', Model_save_path)
#shutil.copy2(Model_save_path+'/range_input.dat', Model_save_path)


# model.save_weights('my_model_weights.h5')

sys.exit()
