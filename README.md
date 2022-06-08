# Double-panel overlay visualizer

#### A python class for visualizing per-pixel overlay in to left/right-panel view.

A tool/class for double-panel visualization (left/right) of multiple overlays displayed over an image. Features:
 - switch between multiple overlay (multiple channels) in left and right panel independently 
 - move between multiple images in folder
 - toggle overlay and change alpha-mixing
 - optionally load and display points
 - user configurable (define main image filename suffix and suffix for left/right overlay filename)


##### Installation and dependencies:

Install as pip package:
```bash
pip install git+https://github.com/vicoslab/dl_visualizer.git
```

Dependencies:
 * Python >= 3.6
 * NumPy >= 1.15.0
 * Python OpenCV >= 4.5

##### Usage:

- Executable as a viewer for pre-generated results (saved as .npy files) 
  ```bash  
  overlay_viz --cfg demo/config.json] --dir <data_folder>
  ```
  ```code
  usage: overlay_viz [-h] [--cfg CONFIG_JSON] [--dir DIR]
  
  Overlay visualization tool.
  
  optional arguments:
    -h, --help  show this help message and exit
    --cfg C     config filename/path
    --dir D     main folder with images (will ask for folder confirmation)
  ```
  
- Class-based usage for a versatile way of viewing the predictions during learning (aka live view) - see `demo/demo.py

##### Config:
Configuration is loaded from the provided `config.json` where you can define:
 - which filename suffix is used for image
 - which filename suffix (one or more) will be used for left and right panel overlays
 - title of left/right panel
 - background colors, loaded center points size, default blending factor, etc

Default configuration is:
```json
{
    "dx": 0.1, 
    "alpha": 0.5,
    "window_name": "image",
    "center_size": 2,
    "center_color": [ 0, 0, 255],
    "panel_title": {
        "left": "GT",
        "right": "pred"
    },
    "file_patterns": {
        "image": "0.img.png",
        "left_pane": ["GT.npy"],
        "right_pane": ["pred.npy"],
        "centers": "centers.npy"
    }
}
```

##### Input formats:
Pre-generated data for visualization can be NumPy arrays (.npy) or images (jpeg/png).

Expected NumPy data should be in the following format:
- rgb image (h x w x 3)
- left/right overlay arrays (c x h x w)
- optional:
    - object centers (n x 2)
  
All data must have the same corresponding size (h x w). Data that is stacked in 2D-grid image will be split into 
multiple patches for display as different channels, each with the same size as the input image.

##### Hotkeys:
    'w/s' - switch left overlay channel index
    'up/down' - switch right overlay channel index
    'left/right' - previous/next image
    'space' - toggle pause (during live view)
    '+/-' - adjust overlay opacity
    't' - toggle right overlay
    'g' - toggle object centers display
    'o' - select directory
    'q' - quit

##### Example data:
Download example data from https://unilj-my.sharepoint.com/:f:/g/personal/jmuhovic_fe1_uni-lj_si/Emsoh0lHX-RGpEAEXJaA7ZwBRvxHkHThyE7Hfir95bSr7A?e=sl0aWp
and run:

```bash
unzip visualizer_data.zip

overlay_viz --dir=./visualizer_data
```
