# Adversarial Domain Adaptation Paper Implementations
This repository is a collection of some unsupervised adversarial domain adaptation models and their pre-trained checkpoints.

This repository is inspired by & heavily borrows code from - [
pytorch-domain-adaptation](https://github.com/jvanvugt/pytorch-domain-adaptation).

Currently, 3 popular Adversarial Domain Adaptation models are trained with MNIST and MNIST-M datasets as their source and target datasets. Visualization of activations and test results from target dataset for the 3 models are also provided using Jupyter Notebook. Ongoing work for adding more implementations of latest research papers in domain-adaptation to the repository.

Please follow the installation instructions from the [parent repository](https://github.com/jvanvugt/pytorch-domain-adaptation).


## Note
This work is purely for experimental purposes. If you are using this work, please ensure you cite the original authors of the papers and the [base repository](https://github.com/jvanvugt/pytorch-domain-adaptation) from which this code was borrowed and modified.


## Implemented papers
**Paper**: Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)  
**Link**: [https://arxiv.org/abs/1409.7495](https://arxiv.org/abs/1409.7495)  
**Description**: Negates the gradient of the discriminator for the feature extractor to train both networks simultaneously.  
**Implementation**: [revgrad.py](https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/revgrad.py)

---

**Paper**: Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)  
**Link**: [https://arxiv.org/abs/1702.05464](https://arxiv.org/abs/1702.05464)  
**Description**: Adapts the weights of a classifier pretrained on source data to produce similar features on the target data.  
**Implementation**: [adda.py](https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/adda.py)

---

**Paper**: Wasserstein Distance Guided Representation Learning, Shen et al. (2017)  
**Link**: [https://arxiv.org/abs/1707.01217](https://arxiv.org/abs/1707.01217)  
**Description**: Uses a domain critic to minimize the Wasserstein Distance (with Gradient Penalty) between domains.  
**Implementation**: [wdgrl.py](https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/wdgrl.py)


## Instructions

#### Note: There is no need to download any dataset for training / testing. The required datasets are provided in *datasets* directory. Pre-trained model checkpoints are present in *trained_models* directory.

1. In a Python 3.6 environment, run:
```
$ conda install pytorch torchvision numpy -c pytorch
$ pip install tqdm opencv-python
```
2. Train a model on the source dataset using the notebook
```
$ train_source.ipynb
```
2. Choose an algorithm and pass it the pretrained network, for example, the notebook:
```
$ python revgrad.ipynb
```
4. To test a model, use the below notebook
Note: Uncomment the *MODEL_FILE* line corresponding to the path of the saved pre-trained checkpoint for the model you're testing:
```
$ python test_model.ipynb
```