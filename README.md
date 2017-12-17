# SRGAN
  - An implement of [SRGAN](https://arxiv.org/abs/1609.04802)(Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network) for tensorflow version.
  - In this repo, vgg19 is not used, instead, MSE is ued to train SRResNet. If you want to use vgg19 to calculate the content loss, [here]() you can download model that trained in ImageNet. Then you just need to load to your model when training time.

# Requirements
  - tensorflow 1.3.0
  - python 2.7.12 or python 3.*
  - numpy 1.13.1
  - scipy 0.17.0
  
# Usages
  ## downlaod repo
  - download this repo by the following instruction:
  
        $ git clone https://github.com/nnuyi/SRGAN.git
        $ cd SRGAN
      
  ## download datasets
  - Firstly, you need to make some directories in the root path(in SRGAN directory)
  
        $ mkdir data
        $ cd data
        $ mkdir train
        $ mkdir val
        $ mkdir test   

  ### train data
  - In this repo, I use parts of [ImageNet]() datasets as **train data**, [here]() you can download the datasets that I used. 
  
  - After you have download the datasets, copy ImageNet(here I only use 3137 images) datsets to ***/data/train***, then you have ***/data/train/ImageNet*** path, and training images are stored in ***/data/train/ImageNet***
  
  - I crop image into **256*256 resolution**, actually you can crop them according to your own.
  
  ### val data
  - Set5 dataset is used as **val data**, you can download it [here]().
  
  - After you download **Set5**, please store it in ***/data/val/***, then you have ***/data/val/Set5*** path, and val images are stored in ***/data/val/Set5***
  
  ### test data
  - Set14 dataset is used as **test data**, you can download it [here]().
  
  - After you download **Set14**, please store it in ***/data/test/***, then you have ***/data/test/Set14*** path, and val images are stored in ***/data/test/Set14***
 
  ## training
  
      $ python main.py --is_training=True --is_testing=False
      
  ## testing
  
      $ python main.py --is_training=False --is_testing=True
      
# Experimental Results
## Factor 4(two shuffle layers is used)
  
  | low resolution| high resolution GT| high resolution GEN|
  |:-----------------:|:-----------------:|:-----------------:|
  | ![Alt test](/data/Set14_gt_lr_2.png)| ![Alt test](/data/Set14_gt_hr_2.png)| ![Alt test](/data/Set14_test_hr_2.png)||
  | ![Alt test](/data/Set14_gt_lr_3.png)| ![Alt test](/data/Set14_gt_hr_3.png)| ![Alt test](/data/Set14_test_hr_3.png)||
  | ![Alt test](/data/Set14_gt_lr_4.png)| ![Alt test](/data/Set14_gt_hr_4.png)| ![Alt test](/data/Set14_test_hr_4.png)||
  | ![Alt test](/data/Set14_gt_lr_5.png)| ![Alt test](/data/Set14_gt_hr_5.png)| ![Alt test](/data/Set14_test_hr_5.png)||
  | ![Alt test](/data/Set14_gt_lr_6.png)| ![Alt test](/data/Set14_gt_hr_6.png)| ![Alt test](/data/Set14_test_hr_6.png)||
  | ![Alt test](/data/Set14_gt_lr_7.png)| ![Alt test](/data/Set14_gt_hr_7.png)| ![Alt test](/data/Set14_test_hr_7.png)||
  
## Factor 4(whole test images)

  |sampling image|
  |:-----------------:|
  |![Alt test](/data/Set14_test.png)|
  |256*256 resolution left:GT right:GEN||

# References
  - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network](https://arxiv.org/abs/1609.04802)
  - [Is the deconvolution layer the same as a convolutional layer](https://arxiv.org/pdf/1609.07009.pdf)

# Contacts
  Email:computerscienceyyz@163.com
