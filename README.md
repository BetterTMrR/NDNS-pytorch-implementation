# NDNS: Labeling Prototype-based Target Samples for Improving Semi-supervised Domain Adaptation

## Abstract
A common practice in semi-supervised domain adaptation (SSDA) approaches is to use the few labeled target samples of each class as class prototypes. By aggregating unlabeled features around these prototypes, the goal is to achieve categorically aligned features. However, this poses a challenge for human annotators as it is difficult to determine which samples are most suitable for labeling as class prototypes from a vast amount of unlabeled data. Additionally, this practice imposes a strict requirement for each class to have at least one labeled target sample, thereby placing a significant manual effort burden on annotators. To address this concerns, we propose an active learning-based SSDA framework to automatically select a prespecified number of target samples which, if labeled, can significantly aid the learning process of SSDA. To achieve this goal, we introduce a novel sample query strategy called Non-maximal Degree Node Suppression (NDNS). NDNS first constructs a directed graph using the target data by defining accepted neighbors (ADNs) and acceptive neighbors (AENs). Then, NDNS iteratively performs maximal degree node query and non-maximal degree node removal to select representative and diverse target samples for labeling. We also inject information of the model uncertainty into our query process, which accounts for the low-confidence parts of the target data. Furthermore, we leverage the prediction agreement of AENs to guide their corresponding disturbed node, producing more compact local clusters on the graph. By using the same annotation budget, our proposed NDNS improves existing SSDA methods with significant margins on three benchmarks, clearly demonstrating the effectiveness of NDNS.



## Installation

`pip install -r requirements.txt`


## Data Preparation
Please download [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/), [Office-Home](http://ai.bu.edu/visda-2017/), and [DomainNet](http://ai.bu.edu/M3SDA/) to ./data

## Training
Here is an example of training on Offce-Home benchmark under 1-shot setting.


(1) Pretain a source model,

`python train_source.py --gpu_id 0 --dset office_home --s 0 --net resnet34 --max_epoch 20 
`

(2) To run training on A to C task on benchmark Office-Home using FixMME method,

`python main.py --dset office_home --gpu_id 0 --s 0 --t 1 --shot 1 --net resnet34 --method FixMME --max_epoch 50 --tar_aen --lam1 0.3 --th 0.8
`

(3) To run training on A to C task on benchmark Office-Home using MME method,

`python main.py --dset office_home --gpu_id 0 --s 0 --t 1 --shot 1 --net resnet34 --method MME --max_epoch 50 --tar_aen --lam1 0.3
`


(4) To run training on A to C task on benchmark Office-Home using MCL method,

`python main.py --dset office_home --gpu_id 3 --s 0 --t 1 --shot 1 --net resnet34 --method MCL --max_epoch 50 --T2 1.25 --lambda_cls 0.2 --th 0.95 --lam1 0.1
`

## Acknowledgement
Our code is partially based on [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME), [SHOT](https://github.com/tim-learn/SHOT), and [NRC](https://github.com/Albert0147/SFDA_neighbors) implementations.
