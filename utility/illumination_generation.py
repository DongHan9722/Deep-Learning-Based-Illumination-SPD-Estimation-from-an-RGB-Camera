#%%
import numpy as np
import random
from numpy import asarray
from utility.library import *

def simulate_illumination(channel_intensity,channel_weight):
    '''
    generate one spectral power distribution of one lighting condition

    INPUTS: channel_intensity = [led1,led2,...,ledn], led is row vector with size 1xM, M is bands number 
            channel_weight = [c1,c2,..,cn], c is scalar value from 0-1
    OUTPUTS: spd = spd for one light
    '''
    simulated_spd = []
    for i in range(len(channel_weight)):
        temp = channel_weight[i]*channel_intensity[i]
        simulated_spd.append(temp)
    spd = sum(simulated_spd)
    spd = asarray(spd)/np.max(spd)
    return spd


def illumination_generation(N,channel_intensity,channel_number):
    '''
    generate N spectral power distribution of N lighting condition

    INPUTS: N = 1,2,3,...,N scalar number 
            channel_intensity = [led1,led2,...,ledn], led is row vector with size 1xM, M is bands number 
            channel_number = scalar number, the number of channel of light
    OUTPUTS: illumination_array = all the illumination SPDs
             weights_list = all the corresponding illumination weights
    '''

    illumination_data = []
    weights_list = []
    for i in range(N):
        weights = list(np.round(np.random.rand(1,channel_number).flatten(),2)) # randomlize the weights from 0-1
        spd = simulate_illumination(channel_intensity,weights)
        illumination_data.append(spd)
        illumination_array = np.array(illumination_data)
        weights_list.append(weights)
    return illumination_array, weights_list

def illumination_generation_match_with_telelumen(N,channel_intensity,channel_number,maximum):
    '''
    generate N spectral power distribution of N lighting condition

    INPUTS: N = 1,2,3,...,N scalar number 
            channel_intensity = [led1,led2,...,ledn], led is row vector with size 1xM, M is bands number 
            channel_number = scalar number, the number of actived channel 
    OUTPUTS: illumination_array = all the illumination SPDs
             weights_list = all the corresponding illumination weights
    '''

    """
    excluding the uv&infrared channel 
    """
    # eg: channel_number = 7 # choose valid activated channel
    weights_list = []
    weight_init = [0,0]          # exclude UV channels
    weight_tail = [0,0,0,0,0]    # exclude IR channels
    weight_zero = [0]*(17-channel_number)  # the other channels are set to 0 except the valid activated channel

    illumination_data = []

    while len(weights_list)<N:
        weights = np.round(np.random.uniform(low=0.0, high=maximum, size=channel_number),2).tolist()
        weights.extend(weight_zero)
        random.shuffle(weights)

        weight_init.extend(weights)
        weight_init.extend(weight_tail)

        weights_list.append(weight_init)
 
        spd = simulate_illumination(channel_intensity,weights)
        illumination_data.append(spd)
        illumination_array = np.array(illumination_data)
        weight_init = [0,0]

    return illumination_array,weights_list

def one_d_gaussian(mu,sigma,x):
    """
    generate the 1D gaussian data.

    INPUTS: mu = the mean of the data.
            sigma = the standard deviation of data.

    OUTPUTS: 1D gaussian distribution

    """
    N = np.sqrt(2*np.pi*np.power(sigma,2))
    fac = np.power(x-mu,2)/np.power(sigma,2)
    # return np.exp(-fac/2)/N 
    return np.exp(-fac/2) # keep the intensity from 0 to 1


def unique_illumination_weights_customized(channel_number,scale):
    '''
    generate all possible combination of illumination weights according to channel number of illumination 
    constraints: the number of scale is equal to number of channel_number

    INPUTS: channel_number = 1,2,..,N  the number of channel
           scale = [c1,c2,..,cn]  c is scalar value from 0-1
    OUTPUTS: res = output weights
    '''
    import random
    unique_res = set()
    max_combination = len(scale)**channel_number
    while len(unique_res)<max_combination: # set the combination number, 5 possible weights, 5 channel, then 5^5

        weights = [random.choice(scale) for _ in range(len(scale))] # random select weight from list

        c = tuple(weights)
        unique_res.add(c) # only keep unique combination

    try:
        unique_res.remove(tuple([0]*len(scale))) # remove the all zero case since it means turn off all the channel
    except:
        unique_weights = list(unique_res)
        res = [list(i) for i in unique_weights]
        # print(f"output length: {len(res)}")
    return res


def unique_illumination_weights_generation(channel_number,scale):
    '''
    generalize function: unique_illumination_weights_customized to fit different number of channel_number and scale

    INPUTS: channel_number = 1,2,..,N  the number of channel
           scale = [c1,c2,..,cn]  c is scalar value from 0-1
    OUTPUTS: output = all possible combination of illumination weights
    '''

    from itertools import combinations
    scale_number = len(scale)

    if scale_number == channel_number:
        output = unique_illumination_weights_customized(channel_number,scale)
        print(f"output length: {len(output)}")
        return output

    if scale_number > channel_number:
        comb = list(combinations(scale,channel_number))
        comb_list = [list(_) for _ in comb]

        res = []
        for i in range(len(comb_list)):
            unique_weight = unique_illumination_weights_customized(channel_number,comb_list[i])
            res.append(unique_weight)


        unique_res = set()
        for i in range(len(res)):
            for j in range(len(unique_weight)):
                c = tuple(res[i][j])
                unique_res.add(c) # only keep unique combination

        unique_weights = list(unique_res) # convert set to list
        output = [list(i) for i in unique_weights] 
        print(f"output length: {len(output)}")
        return output

    if scale_number < channel_number:
        import random
        unique_res = set()
        while len(unique_res)<scale_number**channel_number:
            comb = [random.choice(scale) for _ in range(channel_number)]
            c = tuple(comb)
            unique_res.add(c)
        unique_weights = list(unique_res)
        output = [list(i) for i in unique_weights]
        print(f"output length: {len(output)}")
        return output

def illumination_channel_intensity(channel_number):
    '''
    generate illumination SPDs according to the channel number 
    4 - 7 is gaussian estimation SPDs
    24 is SPDs for 24 channel LED lighting system provided by the company Telelumen
    '''
   
    x = np.linspace(380,780,81)
    '''
    exclude wavelength after 700nm
    '''
    if channel_number == 4:
        y1 = one_d_gaussian(430,15,x)
        y2 = one_d_gaussian(510,20,x)
        y3 = one_d_gaussian(590,20,x)
        y4 = one_d_gaussian(670,15,x)
        led_intensity = [y1,y2,y3,y4]
        
    elif channel_number == 5:
        y1 = one_d_gaussian(430,15,x)
        y2 = one_d_gaussian(490,20,x)
        y3 = one_d_gaussian(550,20,x)
        y4 = one_d_gaussian(610,20,x)
        y5 = one_d_gaussian(670,15,x)
        led_intensity = [y1,y2,y3,y4,y5]

    elif channel_number == 6:

        y1 = one_d_gaussian(420,15,x)
        y2 = one_d_gaussian(470,20,x)
        y3 = one_d_gaussian(520,20,x)
        y4 = one_d_gaussian(570,20,x)
        y5 = one_d_gaussian(620,20,x)
        y6 = one_d_gaussian(670,15,x)
        led_intensity = [y1,y2,y3,y4,y5,y6]

    elif channel_number == 7:

        y1 = one_d_gaussian(430,15,x)
        y2 = one_d_gaussian(470,20,x)
        y3 = one_d_gaussian(510,20,x)
        y4 = one_d_gaussian(550,20,x)
        y5 = one_d_gaussian(590,20,x)
        y6 = one_d_gaussian(630,20,x)
        y7 = one_d_gaussian(670,15,x)
        led_intensity = [y1,y2,y3,y4,y5,y6,y7]
    
    elif channel_number == 24:
        '''
        24 channel illumination SPDs
        '''
        import pandas as pd
        from itertools import islice
        from numpy import asarray
        import luxpy as lx # package for color science calculations 

        df_illumination = pd.read_csv('./data/illumination/telelumen-dittosizer-24.csv',sep = ',')
        # wave = df_illumination['wavelength']
        # illumination_24_channel = []
        # fig, ax = plt.subplots() 
        # for i in range(1,24+1):
        #     temp = list(df_illumination.iloc[:,i])
        #     # illumination_24_channel.append(temp)
        #     ax.plot(wave, temp)
        # plt.title('24 channels illumination SPDs (provided by the company)')
        # plt.show()

        illumination_24c = asarray(df_illumination).T
        ## interpolation and extrapolation 
        illumination_24c_new = lx.cie_interp(illumination_24c, wl_new = np.arange(380,780+1,5), kind = 'S') 
        illumination_24c_new[1:,65:]=0
        led_intensity = [l for l in illumination_24c_new[1:,:]]
    
    for led in led_intensity:
        plt.plot(x,led,linewidth = 3)
        plt.title("selected Leds")
    plt.show()
 
    return led_intensity