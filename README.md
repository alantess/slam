# SLAM Simultaneous localization & mapping  
- Tracks an agent's location in 3D space. 
- Uses a single camera to estimate for localization.
<img src="etc/ptcloud.jpg" /> 

Execute
---
```bash
   python run.py              # Trains the models
```

<p>The run.py accepts the following:</p>

```bash
optional arguments:
  --kitti-dir             Directory for dataset [Kitti Visual Odometry Datasets]
  --img-height            Sets the image height
  --img-width             Sets the image width
  --batch                 Sets the batch size 
  --mode                  Trains needed model / Display Performance
  --video                 Directory or Link (mp4) to driving footage              
```

Directory 
--
    .
    ├── cpp                 # C++ Implementation
    ├── dataset             # Loads the dataset
    ├── etc                 # Junk files
    ├── extract             # Extracts Features
    ├── model_checkpoints   # Holds Models Files
    ├── networks            # Neural Networks
    ├── run.py              # Main Controller
    ├── support             # Helpers functions
    └── vision              # Manages 3D Visualization


Datasets
--
- [Kitti (Depth)](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php)

License
---
MIT


