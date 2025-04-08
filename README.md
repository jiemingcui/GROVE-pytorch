# GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill (CVPR 2025)
<p align="left">
    <a href='https://arxiv.org/abs/2504.04191'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arXiv&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://jiemingcui.github.io/grove/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>


[//]: # (    <a href='https://www.youtube.com/watch?v=QojOdY2_dTQ'>)

[//]: # (      <img src='https://img.shields.io/badge/Video-Youtube-orange?style=plastic&logo=Youtube&logoColor=orange' alt='Video Youtube'>)

[//]: # (    </a>)

[//]: # (    <a href='https://drive.google.com/file/d/1jeFta3iTT7E_m43GpC35FPtE9PdIxdcX/view?usp=sharing'>)

[//]: # (      <img src='https://img.shields.io/badge/Data-Drive-green?style=plastic&logo=Google%20Drive&logoColor=green' alt='Checkpoints'>)

[//]: # (    </a>)



[//]: # (    <a href='https://drive.google.com/file/d/1jeFta3iTT7E_m43GpC35FPtE9PdIxdcX/view?usp=sharing'>)

[//]: # (      <img src='https://img.shields.io/badge/Model-Checkpoints-green?style=plastic&logo=Google%20Drive&logoColor=green' alt='Checkpoints'>)

[//]: # (    </a>)
</p>

[//]: # (<video src="page.mp4" controls="controls" width="1080" height="720"></video>)
![](assets/teaser.png)
**GROVE, a generalized reward framework that enables open-vocabulary pkysical skill leaning without manual engineering or task-specific demonstrations.**

[//]: # (## Introduction)
[//]: # (![]&#40;assets/model.png&#41;)


## TODOs
- [x] Release training and inference code of Pose2CLIP.
- [x] Release well-trained model of Pose2CLIP.
- [x] Release the training data of low-level controller.
- [ ] Release training code of basic RL agents.


### Installation

Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions.

Once Isaac Gym is installed, install the external dependencies for this repo:

```
pip install -r requirements.txt
```

### Training Data

We release all our training motions for low-level controller, which are located in `calm/data/motions/`.Individual motion clips are stored as `.npy` files. Motion datasets are specified by `.yaml` files, which contains a list of motion clips to be included in the dataset. Motion clips can be visualized with the following command:
```
python calm/run.py
--test
--task HumanoidViewMotion
--num_envs 1
--cfg_env calm/data/cfg/humanoid.yaml
--cfg_train calm/data/cfg/train/rlg/amp_humanoid.yaml
--motion_file [Your file path].npy
```

`--motion_file` can be used to visualize a single motion clip `.npy` or a motion dataset `.yaml`.
If you want to retarget new motion clips to the character, you can take a look at an example retargeting script in `calm/poselib/retarget_motion.py`.


## Acknowledgments
Our code is based on [CALM](https://github.com/NVlabs/CALM) and [CLIP](https://github.com/openai/CLIP) and [AnySkill](https://github.com/jiemingcui/anyskill). Thanks for these great projects.

## Citation
```text
@inproceedings{cui2025grove,
  title={GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill},
  author={Cui, Jieming and Liu, Tengyu and Ziyu, Meng and Jiale, Yu and Ran Song and Wei Zhang and Zhu, Yixin and Huang, Siyuan},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

