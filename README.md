# GazeEstimation

This repository is inspired by the paper, [Gaze Estimation using Transformers](https://github.com/yihuacheng/GazeTR) proposed by Cheng et al. 
Please refer to the acknowledgement.


## Preprocessing

The preprocessing has been done using the code from: [Gazehub 3D-dataset](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset//#mpiigaze)

## Usage

Leave-one-person-out evaluation:

```
python trainer/leave.py -s config/train/config_mpii.yaml -p 0
```

Leave-one-person-out training:

```
python trainer/total.py -s config/train/config_mpii.yaml    

```

## Model performance
![Model Performance](./img/Performance.png)

## Citation

```
@InProceedings{cheng2022gazetr,
    title={Gaze Estimation using Transformer},
    author={Yihua Cheng and Feng Lu},
    journal={International Conference on Pattern Recognition (ICPR)},
    year={2022}
}
```

## Acknowledgment
This project is based on and modifies code from the following repository:

- **[GazeTR](https://github.com/yihuacheng/GazeTR)** by Yihua Chen at Beihang University.

The original code is licensed under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/), which allows sharing and adapting the material under certain conditions:
- **Attribution (BY)**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **Non-Commercial (NC)**: You may not use the material for commercial purposes.
- **Share-Alike (SA)**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license.

This modified version remains under the same license.

### Modifications Made
The following modifications have been made to the original code:
- ResNet-18 CNN backbone was replaced by MobileNet V2 and EfficientNet B0

These modifications are shared under the same [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
