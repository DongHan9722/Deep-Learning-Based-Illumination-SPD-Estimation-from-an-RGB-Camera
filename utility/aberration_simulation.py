## resources: https://github.com/SISPO-developers/OASIS
import scipy
import numpy as np
from PIL import Image
def generate_chromatic_aberration(input, amount, width, height):
    '''
    simulation of lateral chromatic aberration

    INPUTS:
        input = image
        amount = the scalar number which indicates the aberration strength
        width = image width
        height = image height

    OUTPUTS:
        input = after adding certain aberration
    '''
    img_r = np.zeros((width, height))
    img_g = np.zeros((width, height))
    img_b = np.zeros((width, height))
    for x in range(0, width):
        for y in range(0, height):
            img_r[x][y] = input[y*width*3 + x*3 + 0]
            img_g[x][y] = input[y*width*3 + x*3 + 1]
            img_b[x][y] = input[y*width*3 + x*3 + 2]
    strength = amount/4000
    img_r = scipy.ndimage.interpolation.zoom(img_r, 1-strength)
    img_b = scipy.ndimage.interpolation.zoom(img_b, 1+strength)
    w_r = len(img_r)
    h_r = len(img_r[0])
    w_b = len(img_b)
    h_b = len(img_b[0])
    if amount > 0:
        for x in range(0, width):
            for y in range(0, height):
                x_r = int(round((w_r-width)/2 + x))
                y_r = int(round((h_r-height)/2 + y))
                x_b = int(round((w_b-width)/2 + x))
                y_b = int(round((h_b-height)/2 + y))
                if 0 <= x_r < w_r and 0 <= y_r < h_r:
                    input[y*width*3 + x*3 + 0] = img_r[x_r][y_r]
                else:
                    input[y*width*3 + x*3 + 0] = 0
                input[y*width*3 + x*3 + 1] = img_g[x][y]
                input[y*width*3 + x*3 + 2] = img_b[x_b][y_b]
    else:
        for x in range(0, width):
            for y in range(0, height):
                x_r = int(round((w_r-width)/2 + x))
                y_r = int(round((h_r-height)/2 + y))
                x_b = int(round((w_b-width)/2 + x))
                y_b = int(round((h_b-height)/2 + y))
                if 0 <= x_b < w_b and 0 <= y_b < h_b:
                    input[y*width*3 + x*3 + 2] = img_b[x_b][y_b]
                else:
                    input[y*width*3 + x*3 + 2] = 0
                input[y*width*3 + x*3 + 1] = img_g[x][y]
                input[y*width*3 + x*3 + 0] = img_r[x_r][y_r]
    del img_r
    del img_g
    del img_b
    print("")
    return input
   
def sRGB_to_linear(sRGB):
    sRGB = sRGB / 255
    if sRGB <= 0.04045:
        return sRGB / 12.92
    else:
        return ((sRGB + 0.055) / 1.055) ** 2.4

def linear_to_sRGB(lin):
    sRGB = 0
    if lin <= 0.0031308:
        sRGB = lin * 12.92
    else:
        sRGB = 1.055 * (lin ** (1 / 2.4)) - 0.055
    return min([255, sRGB * 255])

def read_image(image,dimensions, render_dimensions):

        input = []
        exr_in = 1
        # img = Image.open(file)
        img = image

        input = list(img.getdata())
        width, height = img.size
        input_1d = [0 for i in range(width * height * 3)]
        for i in range(0, width):
            for j in range(0, height):
                for k in range(0, 3):
                    if exr_in == 0:
                        input_1d[i*height*3 + j*3 + k] = sRGB_to_linear(input[i*height + j][k])
                    else:
                        input_1d[i*height*3 + j*3 + k] = input[i*height + j][k]
        del input[:]
        del input
        if render_dimensions[1] - render_dimensions[0] < 1 or render_dimensions[3] - render_dimensions[2] < 1 or render_dimensions[1] > width or render_dimensions[3] > height:
            render_dimensions[0] = 0
            render_dimensions[1] = width
            render_dimensions[2] = 0
            render_dimensions[3] = height
        dimensions[0] = width
        dimensions[1] = height
        print("")
        return input_1d

def write_image(img_array,width, height):
    img = Image.new('RGB', (width, height), (0, 0, 0))
    for i in range(0, width):
        for j in range(0, height):
            ch_r = round(img_array[j*width*3 + i*3 + 0])
            ch_g = round(img_array[j*width*3 + i*3 + 1])
            ch_b = round(img_array[j*width*3 + i*3 + 2])
            img.putpixel((i, j), (int(ch_r), int(ch_g), int(ch_b)))
 
    return img