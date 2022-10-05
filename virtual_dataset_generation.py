#%%
from utility.library import *
from data.read_dataset import read_dataset
from numpy import save, load
from datetime import datetime

#%% 
'''
Loading preprocessed dataset

data type:
    camera sensitivity function: size = 3 x number of bands 
    colorchecker sample: size = number of bands x (number of color patch + 1), the first column is wavelength
    illumination SPDs: size = number of sample x number of bands 

data format standardization:
    wavelength range: [380nm-780nm], 5nm interval
    value range: [0,1]
'''

Cam,illumination,illumination_d65,sample_24,sample_96 = read_dataset() # the default illumination just the testing illumination

#%%
'''
illumination generation (choose Telelumen simulation or gaussian)
'''
from utility.illumination_generation import illumination_generation,illumination_channel_intensity, illumination_generation_match_with_telelumen, unique_illumination_weights_generation

'''
## Telelumen SPD simulation
'''
illumination_intensity = illumination_channel_intensity(24)
illumination, illumination_weights = illumination_generation_match_with_telelumen(100,illumination_intensity,9,0.5)

'''
## gaussian SPD simulation
'''
illumination_intensity = illumination_channel_intensity(5)
illumination, illumination_weights = illumination_generation(100,illumination_intensity,5)

#%%
'''
visualize the simulated illumination 
'''
wave = [i for i in range(380,781,5)] 
fig, axes = plt.subplots(5,2,figsize=(10,10))
plt.text(x=0.5, y=1.55, s="Simulated Illumination", fontsize=18, ha="center", transform=fig.transFigure)
for i, ax in enumerate(axes.flatten()):
    # ax.set_title("{}".format(i))
    ax.plot(wave,illumination[i],linewidth=1.5)
plt.subplots_adjust(top=1.5, wspace=0.3)
plt.show()


#%%
from virtual_camera import virtual_image_computing
#%%
'''
single simulated image computing 
'''
start_time = datetime.now()

'''
parameters
'''

m = 576 # the size of each color patch: 24x24 (this number can be changed to the larger one as long as root(m) is an even integer) 
M = 81
P = 24 
R = 6  
sample = sample_24

sRGB = virtual_image_computing(m,M,P,R,Cam,illumination[2],sample)
plt.imshow(sRGB)
plt.axis('off')
plt.show()
print(sRGB.shape)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

# %%

'''
virtual database generation
'''
start_time = datetime.now()
outputs_sRGB_list = []
fig, axes = plt.subplots(1, 2) 
for i in range(illumination.shape[0]):
    sRGB = virtual_image_computing(m,M,P,R,Cam,illumination[i],sample)
    
    wave = [i for i in range(380,781,5)] 
    axes[0].imshow(sRGB)
    axes[1].plot(wave,illumination[i])
    plt.draw()
    plt.pause(0.0001)
    axes[1].cla()
    print(f"finished {i+1}th")

    # plt.imshow(sRGB)
    # plt.draw()
    # plt.pause(0.001)

    outputs_sRGB_list.append(sRGB)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
# %%
