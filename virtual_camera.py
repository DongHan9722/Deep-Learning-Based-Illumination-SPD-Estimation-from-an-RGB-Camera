#%%
from utility.illumination_generation import illumination_generation
from utility.library import *
from data.read_dataset import read_dataset
from numpy import save, load

# %%
def single_sensor_imaging(m,M,P,R,Cam,illumination,sample):
    '''
    Single sensor camera response computing 
    the formula is inspired from paper "Optimum Spectral Sensitivity Functions for Single Sensor Color Imaging"

    INPUTS:
            varibles:
            m: number of pixels of each color patch (the root of m must be an even integer)
            M: number of bands of illumination SPDs, scene spectral reflectance, camera spectral sensitivity 
            N: number of CFA filter, normally N=3
            P: number of color patch in total
            R: number of color patch in a row
            Cam: camera sensitivity function
            illuminaton: illumination SPDs
            sample: colorchecker spectral reflectance
    denotes:
    A: CFA filter selection matrix
    E: illumination matrix

    OUTPUTS:
            CFA raw image
    '''
    '''
    Bayer pattern computing:
    1. define the element of bayer pattern 
    2. repeat the element to the distinct bayer row according to the length of output pixel block
    3. combine the distinct bayer rows generate from step 2 into one bayer pattern row
    4. repeat bayer pattern row according to total number of pixel  
    '''
    bayer_rg_element = 'rg'
    bayer_gb_element = 'gb'
    block_size = int(math.sqrt(m))

    def repeat_to_length(s, wanted):
        return (s * (wanted//len(s) + 1))[:wanted]

    bayer_rg_row = repeat_to_length(bayer_rg_element,block_size)
    bayer_gb_row = repeat_to_length(bayer_gb_element,block_size)
    bayer_rggb_row = bayer_rg_row + bayer_gb_row
    bayer_rggb_pattern = repeat_to_length(bayer_rggb_row,m)

    '''
    CFA filter selection matrix generation:
    '''
    selected_spectral_sensivity = []
    for i in range(m):
        if bayer_rggb_pattern[i] == 'r':
            selected_spectral_sensivity.append(Cam[0])
        elif bayer_rggb_pattern[i] == 'g':
            selected_spectral_sensivity.append(Cam[1])
        elif bayer_rggb_pattern[i] == 'b':
            selected_spectral_sensivity.append(Cam[2])

    selected_spectral_sensivity_maxtrix = np.row_stack((selected_spectral_sensivity))

    E = np.zeros((M,M), float)
    np.fill_diagonal(E,illumination)
    FCamE = np.matmul(selected_spectral_sensivity_maxtrix,E)

    '''
    computing whole patches from color checker
    '''

    sample_whole = asarray(sample[:,1:]) # whole colorchecker spectral reflectance data excluding the wavelength information in first column 
    CFA_pixel_list = np.matmul(FCamE,sample_whole)
    patch_list = []
    for i in range(CFA_pixel_list.shape[1]):
        patch = CFA_pixel_list[:,i].reshape((block_size, block_size))
        patch_list.append(patch)

    CFA_block_list = patch_list

    horizontal_block_list = []
    for l in range (0, len(CFA_block_list), R):
        horizontal_block = np.concatenate(CFA_block_list[l:l+R], axis = 1)
        horizontal_block_list.append(horizontal_block)
    CFA_raw_whole = np.concatenate(horizontal_block_list)
    CFA_raw_whole_normalized = (CFA_raw_whole/np.amax(CFA_raw_whole)*255).astype(int)

    '''
    8bit, 10bit, 12bit, 16bit CFA raw value computing from single-sensor imaging
    '''
    CFA_raw_whole_1_sensor_8bit = (CFA_raw_whole/np.amax(CFA_raw_whole)*255).astype(int)
    CFA_raw_whole_1_sensor_10bit = (CFA_raw_whole/np.amax(CFA_raw_whole)*1023).astype(int)
    CFA_raw_whole_1_sensor_12bit = (CFA_raw_whole/np.amax(CFA_raw_whole)*4095).astype(int)
    CFA_raw_whole_1_sensor_16bit = (CFA_raw_whole/np.amax(CFA_raw_whole)*65535).astype(int)

    return CFA_raw_whole_normalized, CFA_raw_whole_1_sensor_16bit

#%%
'''
add demosaicing function
'''
from colour_demosaicing import (
  demosaicing_CFA_Bayer_bilinear,
  demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)
#%%
'''
Function to add camera noise
'''
from utility.camera_noise_simulation import add_camera_noise

# %%
'''
Function to add white balance
'''
from utility.white_balance import gray_world_assumes_white_balance, SoG_white_balance, white_patch

#%%
'''
Function to simulate optical aberration 
'''
from utility.aberration_simulation import read_image, write_image, generate_chromatic_aberration
def chromatic_aberration(white_balanced_demosaiced_raw_image):
    '''
    Adding chromatic aberration 
    The lateral chromatic aberration can be simulated by scaling each color channel 
    to produce an increasing offset along the distance from the image center.

    INPUTS:  white balanced image
    OUTPUTS: white balance image with certain chromatic aberration
    '''
    image_width, image_height = white_balanced_demosaiced_raw_image.shape[1], white_balanced_demosaiced_raw_image.shape[0]
    wb = (white_balanced_demosaiced_raw_image/np.amax(white_balanced_demosaiced_raw_image)*255).astype(int)
    array_to_pil = Image.fromarray(np.uint8(wb))
    input_img = read_image(array_to_pil,[image_width,image_height],[0,0,0,0])
    wb_c_aberration = generate_chromatic_aberration(input_img,20,image_width,image_height)
    output = write_image(wb_c_aberration,image_width,image_height)
    output_narray = np.array(output)
    return output_narray
 
#%% 
'''
CIEXYZ computing
'''
from utility.color_matching import CIE_xyz_1931,CIE_light_sources,spim2XYZ
def RGB_average_matrix(img,m):

    '''
    RGB average matrix computing

    INTPUTS:
            img = white-balanced image
    OUTPUTS:
            RGB_average_matrix = average RGB value for each color patch 
    
    Patch size: 24x24
    Edge: 5
    ROI: 14x14
    '''
    average_R_list = []
    average_G_list = []
    average_B_list = []
    for i in range(0,img.shape[0],int(math.sqrt(m))):
        for j in range(0,img.shape[1],int(math.sqrt(m))):
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
def color_correction_matrix(RGB,XYZ):
    '''
    Color correction matrix computing

    INPUTS: RGB = average RGB matrix (3xN)
            XYZ = measured or computed corresponding CIEXYZ matrix (3xN)
    OUTPUTS: 
            theta: color correction matrix

    Illumination: fixed under D65

    '''
    theta = np.dot(np.dot(XYZ,RGB.T), np.linalg.inv(np.dot(RGB, RGB.T)))
    return theta

#%%
def XYZ_estimation(RGB,T):
    '''
    CIEXYZ estimation 
    
    INPUTS: 
            RGB = average RGB matrix 3xN (after white balance)
            T = color correction matrix 
    OUTPUTS:
            XYZ_estimated = corresponding CIEXYZ value for each patch

    Illumination: arbitary illumination 

    '''
    XYZ_estimated = np.dot(T,RGB)
    return XYZ_estimated

#%%
def XYZ_expension(XYZ_estimated,block_size,channel,P):
    '''
    XYZ expension
    repeat the pixel certain times in order to show it as an image
    it is an prestep for estimating the corresponding sRGB image 

    INPUTS:
            XYZ_estimated = CIEXYZ value of each patch
            block_size = size of color patch
            channel = 0,1,2 corresponding R,G,B
            P = number of color patch in total
    OUTPUTS:
            XYZ_whole = CIEXYZ single channel image
    '''
    if P == 24:
        patch_in_each_row = 6
    if P == 96:
        patch_in_each_row = 12

    XYZ_block_list = []
    for i in range(XYZ_estimated.shape[1]):
        p = []
        p.append(XYZ_estimated[channel,i]) # pixel in one channel  

        p_rep = p * block_size * block_size # repeat each pixel block_size^2 times
        XYZ_block = np.reshape(p_rep, (block_size, block_size)) # reshape to a block image
        XYZ_block_list.append(XYZ_block)

    horizontal_XYZ_block_list = []
    for l in range (0, len(XYZ_block_list), patch_in_each_row): 
        horizontal_XYZ_block = np.concatenate(XYZ_block_list[l:l+patch_in_each_row], axis = 1) # construct every 6 block images in each row 
        horizontal_XYZ_block_list.append(horizontal_XYZ_block)

    XYZ_whole = np.concatenate(horizontal_XYZ_block_list) # concatenate each row vertically
    XYZ_whole_normalized = (XYZ_whole/np.amax(XYZ_whole)*255).astype(int)
    return XYZ_whole

#%%
'''
XYZ to sRGB
'''
from utility.color_matching import XYZ2RGB
def color_correction_matrix_generation(m,M,P,R,Cam,illumination_d65,sample):
    '''
    calculate the color correction matrix for colorchecker with P patches 
    '''
    CFA, CFA16 = single_sensor_imaging(m,M,P,R,Cam,illumination_d65,sample)
    gain = 7.1
    quantum_efficiency = 0.35
    electrons_per_pixel = CFA16 / gain
    photons_per_pixel = (electrons_per_pixel / quantum_efficiency).astype(int)

    adu = add_camera_noise(input_irrad_photons=photons_per_pixel, qe=0.35, sensitivity=7.1,
                     dark_noise=3.0, bitdepth=16, baseline=100)
    rgb_demosaiced_noise = demosaicing_CFA_Bayer_Menon2007(adu, pattern='RGGB')
    rgb_demosaiced_normalized_noise = (rgb_demosaiced_noise/np.amax(rgb_demosaiced_noise)*255).astype(int)
    rgb_demosaiced_normalized_noise[rgb_demosaiced_normalized_noise < 0] = 0
    white_balanced_demosaiced_raw_image = SoG_white_balance(rgb_demosaiced_normalized_noise,p=6)
    white_balanced_demosaiced_raw_image_with_noise_with_aberration = chromatic_aberration(white_balanced_demosaiced_raw_image)

    RGB_matrix = RGB_average_matrix(white_balanced_demosaiced_raw_image_with_noise_with_aberration)
    wave = np.linspace(380,780,81)
    XYZ_matrix = spim2XYZ(sample[:,1:], wave,lsource=illumination_d65)
    T = color_correction_matrix(RGB_matrix,XYZ_matrix)
    return T

#%%
'''
sRGB image computing
'''
'''
parameters:
    m = 576
    M = 81
    when using 24 patch colorchecker:
    P = 24
    R = 6
    sample = sample_24
    when using 96 patch colorchecker:
    P = 96
    R = 12
    sample = sample_96
'''
def virtual_image_computing(m,M,P,R,Cam,illumination,sample):

    if P == 24:
        try:
            T = load('data/color correction matrix/T_D65_24.npy') # load precomputed color correction matrix
        except:
            print('file does not exist')
            print(f'calculate color correction matrix for colorchecker {P}')
            T = color_correction_matrix_generation(m,M,P,R,Cam,sample)
            save('data/color correction matrix/T_D65_24.npy',T)

    elif P == 96:
        try: 
            T = load('data/color correction matrix/T_D65_96.npy') # load precomputed color correction matrix
        except:
            print('file does not exist')
            print(f'calculate color correction matrix for colorchecker {P}')
            T = color_correction_matrix_generation(m,M,P,R,Cam,sample)
            save('data/color correction matrix/T_D65_96.npy',T)
    
    CFA, CFA16 = single_sensor_imaging(m,M,P,R,Cam,illumination,sample)

    gain = 7.1
    read_noise = 3.0
    quantum_efficiency = 0.35
    electrons_per_pixel = CFA16 / gain
    photons_per_pixel = (electrons_per_pixel / quantum_efficiency).astype(int)
    adu = add_camera_noise(input_irrad_photons=photons_per_pixel, qe=0.35, sensitivity=7.1,
                    dark_noise=3.0, bitdepth=16, baseline=100
                    #  rs=np.random.RandomState(seed=1000)
                    )

    # adu_normalized = ((adu/np.amax(adu))*255).astype(int)
    rgb_demosaiced_noise = demosaicing_CFA_Bayer_Menon2007(adu, pattern='RGGB')
    rgb_demosaiced_normalized_noise = (rgb_demosaiced_noise/np.amax(rgb_demosaiced_noise)*255).astype(int)
    rgb_demosaiced_normalized_noise[rgb_demosaiced_normalized_noise < 0] = 0

    white_balanced_demosaiced_raw_image = SoG_white_balance(rgb_demosaiced_normalized_noise,p=6)

    output_narray = chromatic_aberration(white_balanced_demosaiced_raw_image)

    RGB_matrix = RGB_average_matrix(output_narray,m)

    XYZ_estimated = XYZ_estimation(RGB_matrix, T)
  
    X_estimated = XYZ_expension(XYZ_estimated,int(np.sqrt(m)),0,P)
    Y_estimated = XYZ_expension(XYZ_estimated,int(np.sqrt(m)),1,P)
    Z_estimated = XYZ_expension(XYZ_estimated,int(np.sqrt(m)),2,P)

    XYZ_estimated_raw = np.dstack((X_estimated,Y_estimated,Z_estimated))
    XYZ_estimated_image = (XYZ_estimated_raw/np.amax(XYZ_estimated_raw)*255).astype(int)
    sRGB = XYZ2RGB(XYZ_estimated_raw)
    
    return sRGB
