
# Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation

 The source code of our work **"Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation**
![img|center](./img/pipeline.png)


## Installation

To create an environment for training and inference, run the following command:

```conda create --name -f <environment.yml file path>```

## Data Preparation

### SemanticKITTI
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### nuScenes
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──v1.0-trainval
		├──v1.0-test
		├──samples
		├──sweeps
		├──maps

```

## Training for SemanticKITTI
1. modify the config/semantickitti.yaml with your custom settings. We provide a sample yaml for SemanticKITTI
2. train the network by running:
```
python train_cylinder_asym.py
```

## Training for nuScenes
1. modify the config/nuScenes.yaml with your custom settings.
2. train the network by running:
```
python train_cylinder_asym_nuscenes.py
```
## Training for heap
1. modify the config/heap.yaml with your custom settings.
2. train the network by running:
```
python train_cylinder_asym_heap.py
```

## Semantic segmentation Inference for a folder of lidar scans
```
python inference_cylinder.py --lidar-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER
```
If you want to validate with your own datasets, you need to provide labels.
--label-folder is optional
```
python inference_cylinder.py --lidar-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER --label-folder YOUR_LABEL_FOLDER
```
## Plotting
To create an environment for plotting, run the following command:

```conda create --name -f <plot_environment.yml file path>```

To plot the inference results you can run:
```
python plot_predictions.py --pcl-file YOUR_PCL_FILE--label-file YOUR_LABEL_FILE --label-color-map YOUR_LABEL_COLOR_MAP
```
, where the label-color-map could for example be: /config/label_mapping/heap.yaml