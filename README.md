# Deep-Learning-Based-Illumination-SPD-Estimation-from-an-RGB-Camera
This is the tentative code for my master thesis. 


## Overall 

Lighting conditions vary spatially and temporally in real life and it significantly influences the real-time perception of people, especially for low-vision patients. Extended reality (XR) technologies and the modern multiple-channel lighting system give the possibility for the digital simulation of low vision in order to investigate the impact of different lighting conditions.

### Standard and MR imaging process
![](images/perceived%20object.png)

The visual similarity between the scene observed by human eyes and the same scene captured and displayed by the stereoscopic camera as well as the VR headset in any lighting condition should be preserved.

Rendering vivid virtual object in VR scene and simulation of some low-vision effects (eg., cataracts) needs the SPD of illumination.
Therefore, my work focus on predict illumination SPD from images.


## Dataset Generation 
There is no existing dataset avalibale for our purpose. In order to acquire the data, we set up the real capturing and also implemented a physically-based image synthesis model.

### virtual camera simulation (for image synthesis)
![](images/virtual%20camera.png)
 
### synthesized image generation (different illumination SPDs applied)
https://user-images.githubusercontent.com/80296065/194082431-8cbee45a-3bb8-4ce7-95b8-b80419c2c41f.mp4

### real captured images (from our capturing setting)
![](images/captured%20images.png)
Our proposed dataset available online: [check the dataset](https://www.couleur.org/articles/SITIS2022-WAI/)

## Illumination SPD estimation
### SPD estimation workflow
![](images/SPD%20estimation%20scheme.png)

## File description 

### Datasets files:
#### Measured dataset (all on 96 colorchecker):
* hue2: 3 RGB channel lamp dataset (Philips hue E27 color lamp, professional camera)
* tablet: 24 channel telelumen dataset (telelumen lighting system, Galaxy Tab S7 tablet camera)
* telelumen: the measured dataset (telelumen lighting system, professional camera)

 
#### Virtual dataset:
* 4-7 channel 10000 random: 4 channel simulated gaussian dataset, 10000 samples, on 24 colorchecker  
* 24 channel 10000 random: 24 channel simulated telelumen dataset, 10000 samples, on 24 colorchecker  
* 24 channel 10000 random 96 unspd 7 activated: 24 channel simulated telelumen dataset, only 7 channel activated, on 96 colorchecker
* 24 channel 10000 random 96 unspd: 24 channel simulated telelumen dataset, all channels activated on 96 colorchecker  
* 17_7_9: the virtual dataset which is corresponding to the telelumen measured dataset, on 96 colorchecker  

### Virtual Camera Model files:
* data: illumination, reflectance, camera sensitivity data 
* utility: data preprocessing, ISP implementation.
* virtual_camera.py
* virtual_dataset_generation.py

### Deep learning model files:
* model: pretrained model
* estimation_rg_Conv1D_measured.py : for working on measured datasets (predict PCA features of SPD)
* estimation_rg_Conv1D_virtual.py : for working on virtual datasets (predict PCA features of SPD)
* estimation_rg_Conv1D_naive.py : for testing model on gaussian virtual datasets (predict the channel weights instead of PCA features of SPD)
 

## Thesis link: [check the thesis for detailed information](https://drive.google.com/file/d/1RuFVj-hMuMlr-eeeA-Pli45FrMztTi8U/view)  

