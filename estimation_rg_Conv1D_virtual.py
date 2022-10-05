#%%
import tensorflow as tf
tf.random.set_seed(123)
from matplotlib import axis
from numpy import asarray, average, load, save
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D,Conv1D,MaxPooling2D, Flatten, Dropout,GlobalAveragePooling1D, MaxPooling1D
from keras.models import load_model
from livelossplot import PlotLossesKeras
from sklearn.preprocessing import MinMaxScaler

#%%
'''
load training, testing, validation data
'''
x_train = load('virtual dataset/average data new/17_7_9/x_train.npy')
x_test = load('virtual dataset/average data new/17_7_9/x_test.npy')
x_val = load('virtual dataset/average data new/17_7_9/x_val.npy')

y_train = load('virtual dataset/average data new/17_7_9/with pca/y_train.npy')
y_test = load('virtual dataset/average data new/17_7_9/with pca/y_test.npy')
y_val = load('virtual dataset/average data new/17_7_9/with pca/y_val.npy')

#%%
'''
normalization
'''
for i in range(x_train.shape[0]):
    x_train[i] = x_train[i] / np.amax(x_train[i])

for i in range(x_test.shape[0]):
    x_test[i] = x_test[i] / np.amax(x_test[i])

for i in range(x_val.shape[0]):
    x_val[i] = x_val[i] / np.amax(x_val[i])

#%%
'''
scaling the features
'''
# scale the input features
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# scale the output targets
scaler_label = MinMaxScaler()
scaler_label.fit(y_train)
y_train = scaler_label.transform(y_train)
y_test = scaler_label.transform(y_test)
y_val = scaler_label.transform(y_val)

#%%
def get_model(n_inputs, n_outputs):

  model = Sequential()

  model.add(Conv1D(32,kernel_size=7,activation='relu',input_shape=n_inputs,name="cnn1")),  
  model.add(Conv1D(64,kernel_size=5,activation='relu',name="cnn2")),  
  # model.add(Conv1D(128,kernel_size=3,activation='relu',name="cnn3")),  
  model.add(MaxPooling1D(pool_size=2,name="maxpooling1")),
  model.add(Dropout(0.2,name="dropout1"))
  model.add(Flatten(name="flatten1")),
  model.add(Dense(64, activation='relu',name="dnn1")), 
  # output layer
  model.add(Dense(n_outputs, activation='sigmoid',name="prediction")), 
  model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.MeanSquaredError(),
            # loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])
  return model

#%%
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
n_inputs, n_outputs = x_train.shape[1:], y_train.shape[1]

#%%
'''
define the training configuration
'''
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

# %%
'''
training and testing 
'''
from datetime import datetime
start_time = datetime.now()
model = get_model(n_inputs,n_outputs)
history = train_model(model, x_train, y_train, x_test, y_test,epochs=20,batch_size=64)
results = evaluate_model(model, x_test, y_test, batch_size=64)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print(results)

#%%
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
"""
Inverse PCA to original data
"""
from sklearn.decomposition import PCA
SPDs_array = load('virtual dataset/training data/17_7_9/y_train.npy') 
pca = PCA(n_components=13) # choose the same component number that used in training dataset
pca.fit(SPDs_array)
print("original shape:", SPDs_array.shape)

inverse = scaler_label.inverse_transform(yhat_array) # inverse the MinMax normalization
reconstructed_illumination = pca.inverse_transform(inverse) # inverse pca to original data
reconstructed_illumination[reconstructed_illumination < 0] = 0 # models negative pixel 
ground_truth_illumination = load('virtual dataset/training data/17_7_9/y_val.npy') # load original groundtruth

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
