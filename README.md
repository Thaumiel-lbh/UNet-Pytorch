# UNet-pytorch
## UNet for pytorch1.11.0  
The UNet-Pytorch that received the most stars on GitHub: https://github.com/milesial/Pytorch-UNet encountered various bugs in The new version pytorch. So I rewrote UNet using PyTorch by referring to https://github.com/milesial/Pytorch-UNet.
## Stage results
### Dataset
This code uses PASCAL_VOC_2012 dataset for training.  
When you use your own dataset, please refer to the PASCAL_VOC2012 dataset structure as follow:  
your_datasets_dir  
　　　　|  
　　　　|  
　　　　|--VOCdevkit  
　　　　　　|  
　　　　　　|  
　　　　　　|--VOC2012  
　　　　　　　　　|  
　　　　　　　　　|  
　　　　　　　　　|--JPEGImages  
　　　　　　　　　　　　　|--SegmentationClass  
　　　　　　　　　　　　　|--ImageSets  
　　　　　　　　　　　　　　　　　|  
　　　　　　　　　　　　　　　　　|  
　　　　　　　　　　　　　　　　　|--Segmentation  
### Training
You can use the following commands for training  
`python train.py --data_root your_datasets_dir/`
