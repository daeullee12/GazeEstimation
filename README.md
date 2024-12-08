# GazeEstimation

This repository is inspired by GazeTR:
I have modified their architecture to compare the performance when MobileNet V2 and EfficientNet B0 were used for CNN backbone.


## Preprocessing

The preprocessing has been done using the code from: [Gazehub 3D-dataset](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset//#mpiigaze)

## Usage

Leave-one-person-out evaluation:
python trainer/leave.py -s config/train/config_xx.yaml -p 0

Leave-one-person-out training:
python trainer/total.py -s config/train/config_xx.yaml    

## Model performance
![Model Performance](./img/Performance.png)

## Citetation
@InProceedings{cheng2022gazetr,
  title={Gaze Estimation using Transformer},
  author={Yihua Cheng and Feng Lu},
  journal={International Conference on Pattern Recognition (ICPR)},
  year={2022}
}
