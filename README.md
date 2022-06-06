# Deep learning visualizer

#### A python class for visualizing per-pixel predictions in deep learning.

##### Usage:
- As a viewer for pre-generated results (saved as .npy files) - demo1()
- As a versatile way of viewing the predictions during learning (aka live view) - demo2()

##### Input formats:
Pre-generated results should be in format {index}_img.npy, {index}_GT.npy, {index}_pred.npy, {index}_centers.npy
    
- rgb image (h x w x 3)
- ground truth array (c x h x w)
- prediction array (c x h x w)
- optional:
    - object centers (n x 2)

##### Hotkeys:
    'w/s' - switch ground truth channel index
    'up/down' - switch prediction channel index
    'left/right' - previous/next image
    'space' - toggle pause (during live view)
    '+/-' - adjust overlay opacity
    't' - toggle prediction overlay
    'g' - toggle object centers display
    'o' - select directory
    'q' - quit

##### Config:
Configuration data is loaded from config.json

##### Example data:
https://unilj-my.sharepoint.com/:f:/g/personal/jmuhovic_fe1_uni-lj_si/Emsoh0lHX-RGpEAEXJaA7ZwBRvxHkHThyE7Hfir95bSr7A?e=sl0aWp