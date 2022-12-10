# Deep Variational Inverse Scattering


[![Paper](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2212.04309)
[![PWC](https://img.shields.io/badge/PWC-report-blue)](https://paperswithcode.com/paper/deep-variational-inverse-scattering)

This repository is the official Pytorch implementation of "[Deep Variational Inverse Scattering](https://arxiv.org/abs/2212.04309)".

| [**Project Page**](https://sada.dmi.unibas.ch/en/research/injective-flows)  | 


<p float="center">
<img src="https://github.com/swing-research/U-Flow/blob/main/figures/network.png" width="1000">
</p>



## Requirements
(This code is tested with pytorch 1.12.1, Python 3.8.3, CUDA 11.0 and cuDNN 7.)
- numpy
- scipy
- matplotlib
- torch==1.12.1

## Installation

Run the following code to install all pip packages:
```sh
pip install -r requirements.txt 
```

## Datasets
We used a synthetic datasets composed of 128x128 images with random ellipses. We added 30dB noise to the measurements and consider two setups for configurations of receivers and incident waves 1) full: where the sensors are uniformy distributed around the object 2) limited-view: where the sensors are located on the right side of the object. You can download the [full](https://drive.switch.ch/index.php/s/NsrsJpzEUpHegfl) and [limited-view](https://drive.switch.ch/index.php/s/2IQIdeWacxSrj6S) datasets and unzip them on the dataset folder.

## Experiments
This is an example of how training the model for 150 epoch for unet and 150 epochs for conditional flow models with limited-view configuration:
```sh
python3 train.py --epochs_unet 150 --epochs_flow 150 --batch_size 64 --dataset scattering --lr 0.0001 --gpu_num 0 --remove_all 0 --desc default --input_type limited-view --train_unet 1 --train_flow 1 --restore_flow 1
```
Each argument is explained in detail in utils.py script.


## Citation
If you find the code or our dataset useful in your research, please consider citing the paper.

```
@article{Khorashadizadeh2022DeepVI,
  title={Deep Variational Inverse Scattering},
  author={AmirEhsan Khorashadizadeh and Ali Aghababaei and Tin Vlavsi'c and Hieu Nguyen and Ivan Dokmani'c},
  journal={arXiv preprint arXiv:2212.04309},
  year={2022}
}
```
