# semi-supervised-learning

This repository implements a collection of Semi-Supervised Learning (SSL) algorithms and evaluates them on Uncertainty Estimation. Algorithms (will) include:
* Mean Teacher (https://arxiv.org/abs/1703.01780)
* Virtual Adverserial Training (https://arxiv.org/abs/1704.03976)
* MixMatch (https://arxiv.org/abs/1905.02249)
* Unsupervised Data Augmentation (https://arxiv.org/abs/1904.12848)


#
### Training a model
Use the following command to train a WideResNet (28-2) with virtual adverserial training
```
python train.py -a wrn --loss vat --checkpoint checkpoints/cifar10/wrn
```

### Evaluating a model
Use the following command to evaluate the previous wrn on detecting out-of-distribution inputs:
```
python detection.py -a wrn --checkpoint checkpoints/cifar10/wrn
```