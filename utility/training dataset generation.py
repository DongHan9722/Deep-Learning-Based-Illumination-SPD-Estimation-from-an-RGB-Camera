#%%
from numpy import asarray, average, load, save
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn
from sklearn.preprocessing import StandardScaler
#%%
'''
load the original dataset
'''
sRGB = load('../measured dataset/original dataset/tablet/pixels_average_rgb_tablet.npy')
illumination = load('../measured dataset/original dataset/tablet/SPDs_tablet.npy')

#%%
'''
splitting dataset for training, testing and validation
'''
train_ratio = 0.70
test_ratio = 0.20
validation_ratio = 0.10
x_train, x_test, y_train, y_test = train_test_split(sRGB, illumination, test_size=test_ratio, random_state=42, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_ratio/(train_ratio+test_ratio), random_state=42, shuffle=True)

#%%
'''
save training, testing, validation data
'''
save('../measured dataset/average data/tablet/original/x_train',x_train)
save('../measured dataset/average data/tablet/original/y_train',y_train)
save('../measured dataset/average data/tablet/original/x_test',x_test)
save('../measured dataset/average data/tablet/original/y_test',y_test)
save('../measured dataset/average data/tablet/original/x_val',x_val)
save('../measured dataset/average data/tablet/original/y_val',y_val)

# %%
"""
Dimension reduction on SPDs
"""
pca = PCA(n_components=10)
pca.fit(y_train)
features_pca_y_train = pca.transform(y_train)
features_pca_y_test = pca.transform(y_test)
features_pca_y_val = pca.transform(y_val)

print("original shape:   ", y_train.shape)
print("transformed shape:", features_pca_y_train.shape)
print("transformed shape:", features_pca_y_test.shape)
print("transformed shape:", features_pca_y_val.shape)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))

save('../measured dataset/average data/tablet/pca/y_train.npy',features_pca_y_train)
save('../measured dataset/average data/tablet/pca/y_test.npy',features_pca_y_test)
save('../measured dataset/average data/tablet/pca/y_val.npy',features_pca_y_val)

