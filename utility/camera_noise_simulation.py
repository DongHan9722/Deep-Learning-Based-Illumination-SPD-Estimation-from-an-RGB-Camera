## resources: http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/
'''
camera noise simulation
'''
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

'''
Photons estimation:
0. Substract the offset value
1. Convert grey values to electrons
2. Convert electrons values to photons

gain 
read noise 
quantum efficiency 
'''
"""
eg:
gain = 7.1
read_noise = 3.0
quantum_efficiency = 0.35
electrons_per_pixel = CFA16 / gain
photons_per_pixel = (electrons_per_pixel / quantum_efficiency).astype(int)
adu = add_camera_noise(input_irrad_photons=photons_per_pixel, qe=0.35, sensitivity=7.1,
                     dark_noise=3.0, bitdepth=16, baseline=100,
                     rs=np.random.RandomState(seed=1000))
"""

# Function to add camera noise
def add_camera_noise(input_irrad_photons, qe=0.69, sensitivity=5.88,
                     dark_noise=2.29, bitdepth=12, baseline=100,
                     rs=np.random.RandomState(seed=1000)):
 
    # Add shot noise
    photons = rs.poisson(input_irrad_photons, size=input_irrad_photons.shape)
    
    # Convert to electrons
    electrons = qe * photons
    
    # Add dark noise
    electrons_out = rs.normal(scale=dark_noise, size=electrons.shape) + electrons
    
    # Convert to ADU and add baseline
    max_adu     = int(2**bitdepth - 1)
    adu         = (electrons_out * sensitivity).astype(int) # Convert to discrete numbers
    adu += baseline
    adu[adu > max_adu] = max_adu # models pixel saturation
    
    return adu

