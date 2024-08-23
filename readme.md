# GGD and its implementation in PyTorch

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Parameters](#parameters)
- [Functions](#functions)
- [Example on Logistic Regression](#example-on-logistic-regression)
- [Example on Linear Regression](#example-on-linear-regression)
- [Example on Training Neural Network](#example-on-training-neural-network)

## Introduction

**GGD** (Grafting Gradient Descent) is initially described in our paper, which replaces common sampling with (without) replacement with importance resampling which successively implements importance sampling on a batch of subsampled sets, and constructs the **grafting gradient** based on the results of importance resampling. Empirical results show that GGD-based methods can achieve comparable results on simple tasks such as logistic regression and linear regression, and these results also implies the usefulness and potential of GGD-based methods in training complicated neural networks when traditional SGD with importance sampling does not work. The detailed experiment results and their analysis are provided in Section 6 of our original paper.

## Usage

With proper datasets downloaded from <a href="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/">Libsvm</a>, <a href="https://archive.ics.uci.edu/dataset/332/online+news+popularity">UCI</a> or Pytorch using codes below, 

```python
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
CIFAR_train = torchvision.datasets.CIFAR10(root='.test/', train = True, download = True, transform = ToTensor())
CIFAR_test = torchvision.datasets.CIFAR10(root='.test/', train = False, download = True, transform = ToTensor())
MNIST_train = torchvision.datasets.MNIST(root='test/', train = True, download = True, transform = ToTensor()) 
MNIST_test = torchvision.datasets.MNIST(root='test/', train = False, download = True, transform = ToTensor()) 
```
user can obtain different experiment results when running different notebook files. For example, running code cells inside the ''GGD_on_logistic_regression_IJCNN.ipynb'' sequentially, user can obtain the experiment results for training logistic regression on IJCNN dataset with GGD with diminishing stepsize sequence, GGD-Adam and GGD-SVRG respectively.

## Parameters

In this section, we introduce some common parameters that are used throughout all these experiments.

- `n`: int.
Training set size. 
- `n1`: int.
Testing set size.
- `weight_decay` or `nabla`:float.
Penalty parameter if $L_{2}$-regularization is used.
- `rbs` or `rec`:int.
Batch size that used by recording dataloaders which exclusively help calculate the train loss, test loss and full gradient norm after one epoch of training.
- `epoches`:int.
Total number of effective passess.
- `b`:int.
Size of batch of subsampled sets. It is set to be 2 as default.
- `m`:int.
Size of subsampled set. It plays a similar role as batchsize in mini-batch SGD.
- `q`:int.
Update period (frequency) used by SVRG-like variance reduction methods.
- `lr_schedule`:string.
Learning rate scheduling for different experiments. Either `constant` or `t-inverse` for logistic regression and linear regression. Only `Poly` for neural network training.
- `lr0`:float.
Fixed learning rate or initial learning rate if `lr_schedule` is `constant` or `t-inverse`.
- `ilr0`:float.
Initial learning rate if `lr_schedule` is `Poly`.
- `flr0`:float.
Final learning rate if `lr_schedule` is `Poly`.

## Functions

In this section, we introduce some common functions that are used throughout all these experiments.

- `LIBSVMdataset`
Read the datasets downloaded from Libsvm. It is worth noting that due to the different format, `LIBSVMdataset` is modified for experiments in IJCNN so that the data samples can be correctly preprocessed.
- `zero_grad`
Reset the gradients of all model parameters.
- `total_loss`
Calculate the train losses for visualization.
- `test_loss`
Calculate the test losses for visualization.
- `total_grad`
Calculate the $L_{2}$-norm of the full gradient for visualization.
- `full_grad`
Calculate the full gradient as required by SVRG-like variance reduction methods.
- `DeviceDataLoader`
Move the pre-defined dataloaders to the specified devices (GPU or CPU).

## Example on Logistic Regression

We solve logistic regression problems on four different datasets, IJCNN, A9A, COVTYPE and RCV1. It is worth noting that if 
other datasets have been successfully loaded as DataLoader in PyTorch, by setting the proper value of parameter `d`, which equals to the features number of training set, we can actually solve any logistic regression with any dataset using GGD-based methods. 

## Example on Linear Regression

''GGD_on_linear_regression_ONR.ipynb'' notebook file actually gives one illustrative example on how to training linear model using GGD-based methods. User can alter some other datasets and manually set the dimension parameter `d` so that these code cells can be used to solve any $L_{2}$-regularized linear regression problem with proper setting of hyperparameters.

## Example on Training Neural Network

Two illustrative but not exhaustive examples are provided here. Users can load their own datasets and training any neural network as they want using the code from ''GGD_on_neural_network_CIFAR10.ipynb'' or ''GGD_on_neural_network_MNIST.ipynb'' as the reference. 
