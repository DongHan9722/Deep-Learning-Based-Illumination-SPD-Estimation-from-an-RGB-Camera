#%%
from matplotlib import axis
from numpy import asarray, average, load, save
import numpy as np
from math import sqrt
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D,Conv1D,MaxPooling2D, Flatten, Dropout,GlobalAveragePooling2D, MaxPooling1D
from keras.models import load_model
from livelossplot import PlotLossesKeras
import pandas as pd
from itertools import islice
from numpy import asarray
import luxpy as lx # package for color science calculations 

#%%
'''
load training, testing, validation data
'''
x_train = load('virtual dataset/average data new/24 channel 10000 random/x_train.npy')
y_train = load('virtual dataset/average data new/24 channel 10000 random/y_train.npy')
x_test = load('virtual dataset/average data new/24 channel 10000 random/x_test.npy')
y_test = load('virtual dataset/average data new/24 channel 10000 random/y_test.npy')
x_val = load('virtual dataset/average data new/24 channel 10000 random/x_val.npy')
y_val = load('virtual dataset/average data new/24 channel 10000 random/y_val.npy')

#%%
'''
define the tentative model 
'''
def get_model(n_inputs, n_outputs):

  model = Sequential()

  model.add(Dense(32, activation='relu',input_shape=n_inputs)),  
  model.add(Dense(32, activation='relu')), 
    
  model.add(Dense(64, activation='relu')), 
  model.add(Dense(64, activation='relu')), 

  model.add(Dense(128, activation='relu')), 
  model.add(Dense(128, activation='relu')), 
  model.add(Dense(256, activation='relu')), 
  model.add(Dense(256, activation='relu')), 
  model.add(Dense(512, activation='relu')), 
  model.add(Dense(512, activation='relu')), 

  model.add(Dropout(0.5)),

  # output layer
  model.add(Dense(n_outputs, activation='sigmoid')),  
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])
  return model

#%%
def get_model(n_inputs, n_outputs):

  model = Sequential()

  model.add(Conv1D(32,kernel_size=3,activation='relu',input_shape=n_inputs)),  
  # model.add(BatchNormalization()),

  model.add(Conv1D(32,kernel_size=3,activation='relu')),  
  # model.add(BatchNormalization()),

  model.add(Conv1D(64,kernel_size=3,activation='relu')),  
  # model.add(BatchNormalization()),

  model.add(Conv1D(64,kernel_size=3,activation='relu')),  
  # model.add(BatchNormalization()),

  model.add(MaxPooling1D(pool_size=2)),
  model.add(Flatten()),

  # model.add(Dropout(0.5)),

  model.add(Dense(512, activation='relu')), 
  model.add(Dense(n_outputs, activation='sigmoid')),  
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])
  return model

#%%
def train_model(model, train_data, train_labels,test_data, test_labels,epochs, batch_size):
  history = model.fit(train_data, train_labels, 
                      batch_size=batch_size, 
                      epochs=epochs, 
                      verbose=1, 
                      validation_data=(test_data, test_labels),
                      callbacks=[PlotLossesKeras()])
  return history


def evaluate_model(model, test_data, test_labels, batch_size):
  results = model.evaluate(test_data, test_labels, batch_size, verbose=1)
  return results

#%%
'''
if using the Conv1D model
'''
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
n_inputs, n_outputs = x_train.shape[1:], y_train.shape[1]

#%%
'''
if using the DNN model
'''
n_inputs, n_outputs = x_train.shape[1:], y_train.shape[1]

# %%
'''
training and testing 
'''
from datetime import datetime
start_time = datetime.now()
model = get_model(n_inputs,n_outputs)
history = train_model(model, x_train, y_train, x_test, y_test,epochs=200,batch_size=64)
results = evaluate_model(model, x_test, y_test, batch_size=64)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

# %%
'''
prediction
'''
yhat_array = model.predict(x_val)

#%%
'''
prediction accuracy
'''
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_val, yhat_array), 4)) 
print("Mean squared error =", round(sm.mean_squared_error(y_val, yhat_array), 4)) 
print("Root mean squared error =", round(sm.mean_squared_error(y_val, yhat_array, squared=False),4))
print("Median absolute error =", round(sm.median_absolute_error(y_val, yhat_array), 4)) 

# %%
'''
reconstruct illumination 
'''
from utility.illumination_generation import simulate_illumination,illumination_generation,illumination_channel_intensity
def illumination_reconstruction(channel_intensity,predicted_weights):
    illumination_data = []
    for weight in predicted_weights:
        spd = simulate_illumination(channel_intensity,weight)
        illumination_data.append(spd)
        np_array = np.array(illumination_data)
    return np_array

#%%
'''
if using the gaussian dataset, choosing the correct number of channels below
'''
illumination_intensity = illumination_channel_intensity(5)

#%%
'''
if using Telelumen datasets
'''
illumination_intensity = illumination_channel_intensity(24)

#%%
'''
compute predicted illumination and get ground truth illumination
'''
reconstructed_illumination = illumination_reconstruction(illumination_intensity, yhat_array)
ground_truth_illumination = illumination_reconstruction(illumination_intensity, y_val)

#%%
"""
visualization of prediction
"""
wave = [i for i in range(380,781,5)] 
fig, axes = plt.subplots(5,2,figsize=(10,10))
plt.text(x=0.5, y=1.55, s="Reconstructed Illumination", fontsize=18, ha="center", transform=fig.transFigure)
for i, ax in enumerate(axes.flatten()):
    # ax.set_title("{}".format(i))
    ax.plot(wave,ground_truth_illumination[i],'k',linewidth=1.5,label='Ground truth')
    ax.plot(wave,reconstructed_illumination[i],'r--',linewidth=1.5, label='Prediction')
    ax.legend(loc ="upper right",prop={'size': 8})
plt.subplots_adjust(top=1.5, wspace=0.3)
plt.show()

#%%
'''
absolute error on whole spectrum
|Egt - Erc|
'''
ae = ground_truth_illumination - reconstructed_illumination 
ae = np.absolute(ae)
aae = np.average(ae, axis=0)
plt.plot(wave,aae)
plt.ylim(0,1)
plt.title('Absolute Average Error')
plt.show()

#%%
'''
RMSE on whole spectrum
'''
print('error on whole spectrum:')
print("Mean absolute error =", round(sm.mean_absolute_error(ground_truth_illumination, reconstructed_illumination), 4)) 
print("Mean squared error =", round(sm.mean_squared_error(ground_truth_illumination, reconstructed_illumination), 4)) 
print("Root mean squared error =", round(sm.mean_squared_error(ground_truth_illumination, reconstructed_illumination, squared=False),4))
print("Median absolute error =", round(sm.median_absolute_error(ground_truth_illumination, reconstructed_illumination), 4)) 

# %%
"""
RMSE, GFC, SAM, SID evaluation mertrics
"""
def GFC(gt,predict):
  upper_part = np.matmul(gt,predict)
  lower_part = np.sqrt(np.sum(gt**2)) * np.sqrt(np.sum(predict**2))
  return upper_part/lower_part

GFC_list = []
for i in range(ground_truth_illumination.shape[0]):
  GFC_test = GFC(ground_truth_illumination[i],reconstructed_illumination[i])
  GFC_list.append(GFC_test)
plt.plot(GFC_list)
plt.show()

import pysptools.distance
SAM_list = []
SID_list = []
RMSE_list = []
for i in range(ground_truth_illumination.shape[0]):
  sam = pysptools.distance.SAM(ground_truth_illumination[i], reconstructed_illumination[i])
  sid = pysptools.distance.SID(ground_truth_illumination[i], reconstructed_illumination[i])
  rmse = (round(sm.mean_squared_error(ground_truth_illumination[i], reconstructed_illumination[i], squared=False),4))

  SAM_list.append(sam)
  SID_list.append(sid)
  RMSE_list.append(rmse)

SID_list = [x for x in SID_list if np.isnan(x) == False] # remove nan value

print(f"RMSE Mean:", np.round(np.mean(RMSE_list),4))
print(f"RMSE Std:", np.round(np.std(RMSE_list),4))
print(f"GFC Mean:",np.round(np.mean(GFC_list),4))
print(f"GFC Std:",np.round(np.std(GFC_list),4))
print(f"SAM Mean:",np.round(np.mean(SAM_list),4))
print(f"SAM Std:",np.round(np.std(SAM_list),4))
print(f"SID Mean:",np.round(np.mean(SID_list),4))
print(f"SID Std:",np.round(np.std(SID_list),4))
# %%
