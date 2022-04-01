# UNet-pytorch
## UNet for pytorch1.11.0  
The UNet-Pytorch that received the most stars on GitHub: https://github.com/milesial/Pytorch-UNet encountered various bugs in The new version pytorch. So I rewrote UNet using PyTorch by referring to https://github.com/milesial/Pytorch-UNet.
## How to Start Traing
### 1 Prepare Dataset
This code uses PASCAL_VOC_2012 dataset for training.  
When you use your own dataset, please refer to the PASCAL_VOC2012 dataset structure as follows:  
  　  
     
***your_datasets_dir   
　　　　|--VOCdevkit  
　　　　　　　|--VOC2012  
　　　　　　　　　　|--JPEGImages  
　　　　　　　　　　|--SegmentationClass  
　　　　　　　　　　|--ImageSets  
　　　　　　　　　　　　　　|--Segmentation***  
                   　  
                      
These folders and their subfiles are necessary.  
### 2 Prepare Environment  
```bash
pip install -r requirements.txt
```  
### 3 Quick Training
You can use the following commands for training  
```bash
python train.py --data_root your_datasets_dir/
```
### 4 Training Visualization  
You can visualize you training process by using visdom as follows:  
#### 4.1 Run visdom server on port(for example 8097) in your virtual environment.  
```bash
python -m visdom.server -port 8097
```  
#### 4.2 Enable visdom for your training.  
```bash
python train.py --data_root your_datasets_dir/ --enable_vis --vis_port 8097
```  
#### 4.3 You can see your training process at http://localhost:8097  
