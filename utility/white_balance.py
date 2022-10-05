from skimage import img_as_ubyte
import numpy as np

def gray_world_assumes_white_balance(img):
    """
    gray world assumption

    INPUTS: img = demosaied raw image
    OUTPUTS: wb_img = white balanced image
    """
    R, G, B = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
    R_ave, G_ave, B_ave = np.mean(R), np.mean(G), np.mean(B)
    K = (R_ave + G_ave + B_ave) / 3
    Kr, Kg, Kb = K / R_ave, K / G_ave, K / B_ave
    Ra = (R * Kr)
    Ga = (G * Kg)
    Ba = (B * Kb)

    # for i in range(len(Ba)):
    #     for j in range(len(Ba[0])):
    #         Ba[i][j] = 255 if Ba[i][j] > 255 else Ba[i][j]
    #         Ga[i][j] = 255 if Ga[i][j] > 255 else Ga[i][j]
    #         Ra[i][j] = 255 if Ra[i][j] > 255 else Ra[i][j]
    Ra[Ra > 255] = 255
    Ga[Ga > 255] = 255
    Ba[Ba > 255] = 255

    # print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
    wb_img = np.uint8(np.zeros_like(img))
    wb_img[:, :, 0] = Ra
    wb_img[:, :, 1] = Ga
    wb_img[:, :, 2] = Ba
    return wb_img

def SoG_white_balance(img,p):  
    """
    shade of gray 

    INPUTS: img = demosaied raw image
    OUTPUTS: wb_img = white balanced image
    """
    import copy
    img = img/np.amax(img)
    wb_img = copy.deepcopy(img)
    imP = img**p

    R_avg = np.mean(imP[:,:,0])**(1/p)
    G_avg = np.mean(imP[:,:,1])**(1/p)
    B_avg = np.mean(imP[:,:,2])**(1/p)

    Avg = np.mean(imP)**(1/p)
    
    kr = R_avg/Avg
    kg = G_avg/Avg
    kb = B_avg/Avg

    wb_img[:,:,0] = wb_img[:,:,0]/kr
    wb_img[:,:,1] = wb_img[:,:,1]/kg
    wb_img[:,:,2] = wb_img[:,:,2]/kb

    max_value = 1

    wb_img[wb_img > max_value] = max_value # models pixel saturation

    return wb_img
    
def white_patch(img, percentile=100):
    """
    White patch algorithm

    INPUTS: img = demosaied raw image
            percentile = Percentile value to consider as channel maximum   
    OUTPUTS: wb_img = white balanced image
    """
    white_patch_image = img_as_ubyte((img*1.0 / 
                                   np.percentile(img,percentile,
                                   axis=(0, 1))).clip(0, 1))
    return white_patch_image