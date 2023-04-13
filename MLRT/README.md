
### Preparation

#### Requirements: Python=3.5 and Pytorch=0.4.1

1. Install [Pytorch](http://pytorch.org/)

2. Download Cityscape & Fogy Cityscape & BDD100K dataset
      
### Train and Test

1.Change the dataset root path in ./lib/dataset/dgunionlable.py and some save dir path in ./train.py and ./RST.py

2 Train the model
 ```Shell
 # augmentation
 CUDA_VISIBLE_DEVICES=GPU_ID python RST.py

 # train
 CUDA_VISIBLE_DEVICES=GPU_ID python train.py
 
 # Test model
 CUDA_VISIBLE_DEVICES=GPU_ID python test.py
 ```
 
