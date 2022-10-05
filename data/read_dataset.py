import pandas as pd
from utility.library import *
import luxpy as lx # package for color science calculations 
from numpy import asarray

def read_dataset():
    try:
        df_camera = pd.read_csv('./data/camera/Canon/EOS 5D Mark II mod/EOS 5D Mark II mod_full_bw486_1.csv',sep = ';')
        df_illumination = pd.read_csv('./data/illumination/Illuminants.csv',sep = ',')
        df_sample = pd.read_csv('./data/sample/MacbethColorChecker.csv',sep = ',')
    except FileNotFoundError as err:
        print('file does not exist')
        raise err

    wave = df_illumination['l']
    wave = asarray(wave)
    illumination_d65 = df_illumination['D65']
    illumination_d65 = asarray(illumination_d65)/np.max(illumination_d65)
    illumination_d50 = df_illumination['D50']
    illumination_d50 = asarray(illumination_d50)/np.max(illumination_d50)
    illumination_A = df_illumination['A']
    illumination_A = asarray(illumination_A)/np.max(illumination_A)
    illumination = np.row_stack((illumination_d50,illumination_d65,illumination_A))


    plt.plot(wave,illumination_d65,'r',label = 'D65')
    plt.plot(wave,illumination_d50,'g', label = 'D50')
    plt.plot(wave,illumination_A,'b',label='A')
    plt.title("Test illumination")
    plt.legend()
    plt.show()



    sample_24 = asarray(df_sample.iloc[:,0:25]) 

    ## Akima algorithm to interpolate the missing values
    wave_cam = df_camera['Lambda']
    wave_cam = asarray(wave_cam[0:41])
    sensitivity_r = df_camera['R_SR']
    sensitivity_g = df_camera['G_SR']
    sensitivity_b = df_camera['B_SR']
    sensitivity_r = asarray(sensitivity_r[0:41])
    sensitivity_g = asarray(sensitivity_g[0:41])
    sensitivity_b = asarray(sensitivity_b[0:41])
    interp_length = np.arange(wave_cam[0], wave_cam[-1]+5, 5)
    sensitivity_r_interp = Akima1DInterpolator(wave_cam, sensitivity_r)(interp_length)
    sensitivity_g_interp = Akima1DInterpolator(wave_cam, sensitivity_g)(interp_length)
    sensitivity_b_interp = Akima1DInterpolator(wave_cam, sensitivity_b)(interp_length)
    Cam = np.row_stack((sensitivity_r_interp,sensitivity_g_interp,sensitivity_b_interp))

    # plt.title('Akima interpolation of camera sensitivity')
    # plt.plot(interp_length, sensitivity_r_interp, 'bx', label='akima', linewidth=2.5)
    # plt.plot(wave_cam, sensitivity_r, 'go', label='data',linewidth=0.1)
    # plt.legend()
    # plt.show()

    ## measured camera’s raw intensity response
    plt.plot(wave,sensitivity_r_interp,'r')
    plt.plot(wave,sensitivity_g_interp,'g')
    plt.plot(wave,sensitivity_b_interp,'b')
    # plt.title("Camera spectral sensitivity with bw46 UV,IR filter after interpolation")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Response")
    plt.show()

    '''
    loading data of the 96 colorchecker
    '''
    from itertools import islice

    res = []
    i = 0
    for _ in range(96):
        with open('data/sample/DCSG-i1Pro2_96.txt') as lines:
                array = np.genfromtxt(islice(lines, 2+i, 38+i))
                res.append(array[:,1])
                i = i + 39

    res_array = asarray(res)
    wave = [i for i in range(380,731,10)] 
    wave = asarray(wave).T
    ref = np.vstack((wave,res_array))

    REFi = lx.cie_interp(ref, wl_new = np.arange(380,780+1,5), kind = 'S', extrap_kind='linear',extrap_log=True) 
    # print('* REFi.shape --> (M + 1 x number of wavelengths): {}'.format(REFi.shape))
    # lx.SPD(ref).plot(linestyle='solid')
    # lx.SPD(REFi).plot(linestyle='dotted', linewidth  = 2)
    # plt.xlim([380,780])
    sample = REFi.T 

    """
    order the reflectance of each color patch by row-wise
    """
    num_row = 1
    index = 1
    i = 1
    patch_list = [sample[:,0]]

    while num_row <= 8:
        for _ in range(12):
            patch_list.append(sample[:,i])
            i += 8
        num_row += 1
        index += 1
        i = index

    sample_96 = np.array(patch_list).T

    return Cam,illumination,illumination_d65,sample_24, sample_96, 