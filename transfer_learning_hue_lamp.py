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
import sklearn
from sklearn.decomposition import PCA

#%%
'''
load dataset
'''
x_train = load('measured dataset/average data/hue2/pca/x_train.npy')
y_train = load('measured dataset/average data/hue2/pca/y_train.npy')
x_test = load('measured dataset/average data/hue2/pca/x_test.npy')
y_test = load('measured dataset/average data/hue2/pca/y_test.npy')
x_val = load('measured dataset/average data/hue2/pca/x_val.npy')
y_val = load('measured dataset/average data/hue2/pca/y_val.npy')

#%%
'''
data preprocessing
'''
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_val = x_val.astype('float32')

# x_train /= 255
# x_test /= 255
# x_val /= 255

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
n_inputs, n_outputs = x_train.shape[1:], y_train.shape[1]

#%%
'''
define the model
'''
def get_model(n_inputs,n_outputs):
  model = Sequential()

  model.add(Conv1D(32,kernel_size=7,activation='relu',input_shape=n_inputs,name="cnn1")),  
  model.add(Conv1D(64,kernel_size=5,activation='relu',name="cnn2")),  
  model.add(MaxPooling1D(pool_size=2,name="maxpooling1")),
  model.add(Dropout(0.5,name="dropout1"))
  model.add(Flatten(name="flatten1")),
  model.add(Dense(64, activation='relu',name="dnn1")), 
  # output layer
  model.add(Dense(n_outputs, activation='sigmoid',name="prediction")), 

  return model

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
"""
load pretrianed model
"""
pre_model = get_model(n_inputs,24) # get pre-trained model, n_outputs should be set to euqal to previous trained output
pre_model.load_weights("model/pretrained_weights_24_10000_96_unspd_in_right_order.h5")

#%%
"""
configure the new model based on pretrianed model
"""
extracted_layers = pre_model.layers[:-1]
extracted_layers.append(keras.layers.Dense(n_outputs, activation='sigmoid',name="dnn2"))
model = keras.Sequential(extracted_layers)
model.summary()

# %%
'''
training and testing 
'''
from datetime import datetime
start_time = datetime.now()

# model = get_model(n_inputs,n_outputs) # uncomment it if not use transfer learning
model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
          loss=tf.keras.losses.MeanSquaredError(),
          # loss=tf.keras.losses.MeanAbsoluteError(),
          metrics=[tf.keras.metrics.RootMeanSquaredError()])

history = train_model(model, x_train, y_train, x_test, y_test,epochs=100,batch_size=64)
results = evaluate_model(model, x_test, y_test, batch_size=64)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print(results)

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
"""
Inverse PCA to original data
"""
SPDs_array = load('measured dataset/original dataset/hue2/SPDs.npy')
mu = np.mean(SPDs_array, axis=0)
pca = PCA(n_components=3)
pca.fit(SPDs_array)
# features_pca = pca.transform(SPDs_array)
print("original shape:", SPDs_array.shape)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))

reconstructed_illumination = pca.inverse_transform(yhat_array) # inverse pca to original data
reconstructed_illumination[reconstructed_illumination < 0] = 0 # models negative pixel 
ground_truth_illumination = load('measured dataset/average data/hue2/original/y_val.npy') # load original groundtruth

# %%
'''
RMSE of whole spectrum
'''
print('error on whole spectrum:')

print("Mean absolute error =", round(sm.mean_absolute_error(ground_truth_illumination, reconstructed_illumination), 4)) 
print("Mean squared error =", round(sm.mean_squared_error(ground_truth_illumination, reconstructed_illumination), 4)) 
print("Root mean squared error =", round(sm.mean_squared_error(ground_truth_illumination, reconstructed_illumination, squared=False),4))
print("Median absolute error =", round(sm.median_absolute_error(ground_truth_illumination, reconstructed_illumination), 4)) 

# %%
"""
visualization of prediction
"""
wave = [i for i in range(380,731,10)] 
fig, axes = plt.subplots(5,2,figsize=(10,10))
plt.text(x=0.5, y=1.55, s="Reconstructed Illumination", fontsize=18, ha="center", transform=fig.transFigure)
for i, ax in enumerate(axes.flatten()):
    # ax.set_title("{}".format(i))
    ax.plot(wave,ground_truth_illumination[i],'k',linewidth=1.5)
    ax.plot(wave,reconstructed_illumination[i],'r--',linewidth=1.5)
plt.subplots_adjust(top=1.5, wspace=0.3)
plt.show()

# %%
"""
RMSE, GFC, SAM, SID evaluation mertrics

Each element of the SAM score is a spectral angle in radians in the range [0, 3.142]. 
A smaller SAM score indicates a strong match between the test signature and the reference signature.
The smaller the divergence, the more likely the pixels are similar.
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
