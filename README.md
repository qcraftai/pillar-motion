## Self-Supervised Pillar Motion Learning for Autonomous Driving
<p align='left'>
  <img src='example.gif' width='675'/>
</p>

[Chenxu Luo](https://chenxuluo.github.io/), [Xiaodong Yang](https://xiaodongyang.org/), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/) <br>
Self-Supervised Pillar Motion Learning for Autonomous Driving, CVPR 2021<br>
[[Paper]](https://arxiv.org/pdf/2104.08683.pdf) [[Poster]](poster.pdf) [[YouTube]](https://youtu.be/Y00ujpmauUU)

## Getting Started
### Installation
Install [PyTorch](https://pytorch.org/), [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [Apex](https://github.com/NVIDIA/apex), [nuScenes Devkit](https://github.com/nutonomy/nuscenes-devkit)

### Data Preparation
```
python tools/create_data nuscenes_data_prep --root_path /path/to/nuscenes 
```
Our optical flow model used for the cross-sensor regularization is available [here](https://drive.google.com/file/d/1VuCDLcUTYLyON12v81vGN14igG5TfR-j/view?usp=sharing).

### Training
```
python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py configs/nusc_pillarmotion.py --work_dir experiments/pillarmotion/
```



## Citation
Please cite the following paper if this repo helps your research:
```bibtex
@InProceedings{Luo_2021_CVPR,
    author    = {Luo, Chenxu and Yang, Xiaodong and Yuille, Alan},
    title     = {Self-Supervised Pillar Motion Learning for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3183-3192}
}
```

## License
Copyright (C) 2021 QCraft. All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (Attribution-NonCommercial-ShareAlike 4.0 International). The code is released for academic research use only. For commercial use, please contact [business@qcraft.ai](business@qcraft.ai).
