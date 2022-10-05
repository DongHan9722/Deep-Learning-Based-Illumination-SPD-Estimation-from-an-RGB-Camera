#%%
from tqdm import tqdm
import numpy as np
import random

# %%
"""
generate primaries for Philip hue lamp
1. create the txt file according to the path
2. define the channel_number and maximum value
"""
def weights_generation_hue(N,channel_number,maximum):
    '''
    INPUTS: N = number of samples
            channel_number = number of actived channels (3 since it is rgb lamp)
            maximum = maximum intensity of each actived channel
    OUTPUTS:
            the generated weight that needs sending to the lighting system
    '''
    weights_list = []
    while len(weights_list)<N:
        weights = np.round(np.random.uniform(low=0.0, high=maximum, size=channel_number),2) # weights: [0.0-maximum]
        string = " ".join(str(e) for e in weights) # convert list to str [1,2,3] ---> '1,2,3'
        weights_list.append(string)

    for i in tqdm(weights_list):
        f = open('/Users/clyde/Downloads/primaries_hue.txt', 'a')
        # f.write(str(i))
        f.write("all on RGB " + (i) + '\n')
        f.close()

weights_generation_hue(10,3,0.5)
# %%
"""
generate primaries for Telelumen lighting system
(excluding the uv&infrared channel)
"""
def weights_generation_telelumen(N,channel_number,maximum):

    '''
    INPUTS: N = number of samples
            channel_number = number of actived channels (1-17 after excluding the UV&IR channels)
            maximum = maximum intensity of each actived channel
    OUTPUTS:
            the generated weight that needs sending to the lighting system
    '''
    weights_list = []
    weight_init = [0,0]
    weight_tail = [0,0,0,0,0]
    weight_zero = [0]*(17-channel_number)

    while len(weights_list)<N:
        weights = np.round(np.random.uniform(low=0.0, high=maximum, size=channel_number),2).tolist()
        weights.extend(weight_zero)
        random.shuffle(weights)

        weight_init.extend(weights)
        weight_init.extend(weight_tail)

        string = " ".join(str(e) for e in weight_init) # convert list to str [1,2,3] ---> '1,2,3'
        weight_init = [0,0] # reset initial head value to zero
        weights_list.append(string)
    
    print(f"length of each weights:{len(weights)}")
    for i in tqdm(weights_list):
        f = open('/Users/clyde/Downloads/primaries_telelumen.txt', 'a')
        # f.write(str(i))
        f.write("telelumen24 " + (i) + '\n')
        f.close()

weights_generation_telelumen(10,17,0.7)


