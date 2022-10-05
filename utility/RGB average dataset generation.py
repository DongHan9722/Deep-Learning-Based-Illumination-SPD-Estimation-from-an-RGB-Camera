#%%
from numpy import asarray, average, load, save
import numpy as np

#%%
'''
sRGB average matrix computing
'''
def sRGB_average_matrix(img):
    '''
    compute average sRGB value for each color patch for an colorchecker
    ROI has size 14x14 from the center of each color patch

    INPUTS: img = sRGB image with the size of each patch 24x24 
    OUTPUTS: RGB_average_matrix = matrix with size 3x patch number, each column represent sRGB value for each patch
    '''
    average_R_list = []
    average_G_list = []
    average_B_list = []
    for i in range(0,img.shape[0],24):
        for j in range(0,img.shape[1],24):
            # print(i,j)
            R = img[:,:,0]
            G = img[:,:,1]
            B = img[:,:,2]

            block_R = R[i+5:i+5+14,j+5:j+5+14]
            average_R = np.sum(block_R)/(14*14)
            average_R_list.append(average_R)

            block_G = G[i+5:i+5+14,j+5:j+5+14]
            average_G = np.sum(block_G)/(14*14)
            average_G_list.append(average_G)

            block_B = B[i+5:i+5+14,j+5:j+5+14]
            average_B = np.sum(block_B)/(14*14)
            average_B_list.append(average_B)
    RGB_average_matrix = np.row_stack((average_R_list,average_G_list,average_B_list))
    return RGB_average_matrix

#%%
'''
load training, testing, validation data
'''
x_train = load('../virtual dataset/training data/24 channel 10000 random 96 unspd 7 activated/with pca/x_train.npy')
x_test = load('../virtual dataset/training data/24 channel 10000 random 96 unspd 7 activated/with pca/x_test.npy')
x_val = load('../virtual dataset/training data/24 channel 10000 random 96 unspd 7 activated/with pca/x_val.npy')

#%%
'''
normalize if needed
'''
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_test /= 255
x_val /= 255

#%%
"""
compute sRGB average value for training, testing, validation sRGB image dataset
the output store as matrix with size: number of sample x (3x number of patch) 
"""

'''
x_train
'''
sRGB_average_data = []
for i in range(x_train.shape[0]):
    temp = sRGB_average_matrix(x_train[i])
    temp = temp.flatten('F')    # flatten array into 1D [R1,G1,B1,R2,G2,B2,...RN,GN,BN]  
    sRGB_average_data.append(temp)

sRGB_average_data_array_train = asarray(sRGB_average_data)

'''
x_test
'''
sRGB_average_data = []
for i in range(x_test.shape[0]):
    temp = sRGB_average_matrix(x_test[i])
    temp = temp.flatten('F')
    sRGB_average_data.append(temp)

sRGB_average_data_array_test = asarray(sRGB_average_data)


'''
x_val
'''
sRGB_average_data = []
for i in range(x_val.shape[0]):
    temp = sRGB_average_matrix(x_val[i])
    temp = temp.flatten('F')
    sRGB_average_data.append(temp)

sRGB_average_data_array_val = asarray(sRGB_average_data)

# %%
'''
save training, testing, validation data
'''
save('../virtual dataset/average data new/24 channel 10000 random 96 unspd 7 activated/with pca/x_train.npy',sRGB_average_data_array_train)
save('../virtual dataset/average data new/24 channel 10000 random 96 unspd 7 activated/with pca/x_test.npy',sRGB_average_data_array_test)
save('../virtual dataset/average data new/24 channel 10000 random 96 unspd 7 activated/with pca/x_val.npy',sRGB_average_data_array_val)
# %%
