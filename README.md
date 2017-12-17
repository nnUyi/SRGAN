# SRGAN
  - An implement of SRGAN(Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network) for tensorflow version.
  - In this repo, vgg19 is not used, instead, MSE is ued to train SRResNet. If you want to use vgg19 to calculate the content loss, [here]() you can download model that trained in ImageNet. Then you just need to load to your model when training time.

# Requirements
  - tensorflow 1.3.0
  - python 2.7.12 or python 3.*
  - numpy 1.13.1
  - scipy 0.17.0
  
# Usage
  ## downlaod repo
      $ git clone https://github.com/nnuyi/SRGAN.git
      $ cd SRGAN
      
  ## download datasets
    - In this repo, I use parts of [ImageNet]() datasets, [here] you can download the datasets that I used. After you have download the datasets, then **data directory** need to be created as following:
    
      $ mkdir data
      $ cd data
      $ mkdir train
      $ mkdir val
      $ mkdir test   
      
    - Then, copy ImageNet(here I only use 3137 images) datsets to /data/train, then you have /data/train/ImageNet path, and training images are stored in /data/train/ImageNet
    
# Experimental Results

# Contacts
  Email:computerscienceyyz@163.com
