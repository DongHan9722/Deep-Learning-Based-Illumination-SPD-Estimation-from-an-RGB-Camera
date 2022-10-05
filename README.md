# Deep-Learning-Based-Illumination-SPD-Estimation-from-an-RGB-Camera
This is the tentative code for my master thesis 

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
 
## Illustration Figure
 
### Virtual Camera Simulation
![](images/virtual%20camera.png)
 
### Synthesized Image Generation (different illumination SPDs applied)
https://user-images.githubusercontent.com/80296065/194082431-8cbee45a-3bb8-4ce7-95b8-b80419c2c41f.mp4

### SPD Estimation Workflow

## Master thesis link: [clik](https://drive.google.com/file/d/1RuFVj-hMuMlr-eeeA-Pli45FrMztTi8U/view)  
