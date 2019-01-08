# SGAN: several GAN

Pytorch implementation of [SGAN: An Alternative Training of Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chavdarova_SGAN_An_Alternative_CVPR_2018_paper.pdf) where global Discriminator and Generator are trained using local pairs (GANs).

![SGAN_scheme](data/results/SGAN_scheme.png)

## Dependencies

- Python==2.7+
- scipy==1.1.0
- six==1.11.0
- tensorboardX==1.4
- tensorflow==1.4.1
- tensorflow-tensorboard==1.5.1
- torch==0.4.0
- torchvision==0.2.1
- easydict==1.9
- matplotlib==3.0.0
- numpy==1.15.4

## Usage

In config.py you can set up your own parameters: </br>
1. Dataset type.
2. Parameters values for training SGAN.
3. Folders/files name for saving training process/result.

I worked with MNIST and CelebA, for downloading these datasets you can use scripts from this [repo]().

In main.py training process is running.

## Results

#### MNIST

After 1st epoch (128 batch size):
 - Global pair:</br>
![mnist1_global](data/results/mnist/global_pair_epoch_0_batch_45.png)

- Local pair #1:</br>
![mnist1_local1](data/results/mnist/pair1_epoch_0_batch_45.png)

After 14th epoch:</br>
- Global pair:</br>
![mnist14_global](data/results/mnist/global_pair_epoch_14_batch_45.png)

#### CelebA
After 1st epoch (128 batch size):
 - Global pair:</br>
![celebA1_global](data/results/celebA/global_pair_epoch_0_batch_150.png)

- Local pair #5:</br>
![celebA1_local5](data/results/celebA/pair5_epoch_0_batch_150.png)

After 2nd epoch:</br>
- Global pair:</br>
![celebA2_global](data/results/celebA/global_pair_epoch_1_batch_150.png)

Inception Score on validation dataset for Global Pair:
![celebA_IS_global](data/results/celebA/InceptionScore_validation_global_pair.png)

## Training details

There is a possibility to use WGAN and WGAN with gradient penalty,</br>
but I could't succeed with it. If you see any error in code, please let me know!</br>
I achieved such results using DCGAN with vanilla loss function based on Kullback-Leibler Divergence.


## Related works

- [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
- [Inception Score Pytorch](https://github.com/sbarratt/inception-score-pytorch)


## Author

Firiuza Shigapova / [@Firyuza](https://github.com/Firyuza) github / [@SirenaFiriuza](https://medium.com/@SirenaFiriuza) medium
