# Experimental Dependency #
We provide our **pytorch** implementation for CPPN, and all dependencies of our code are listed in "requirements.txt".

## Machine Configurations ##
    Operating System: Win10 
    CPU: Intel Core i9-14700KF
    GPU: NVIDIA Geforce RTX 4090, 24G VRAM
    Memory: 128G
    Hard Disk: 4T SSD

## Code Structure ##
+ **config**: It contains the configuration of hyper-parameters for each dataset to reproduce our corresponding experimental results reported in the paper.
+ **dataset**: This folder contains five zero-shot dataset folders. The sources and brief descriptions of datasets are introduced in the paper, so you can download them and save them into this folder to run our code.
+ **engine**: It consists of an evaluation script and two training modules.
+ **log**: This folder is used to record the training logs.
+ **models**: Our implementations of CPPN is included.
+ **tools**: Some separable modular implementations.

## How to Run ##
We provide the optimal hyperparameter configurations for each dataset, and these can be found in the folder "./config". 
You can use the Python command to run "./main.py" directly with/without cmd arguments what you want to set. 
You can select one of the datasets to run and the details are described in the file "./main.py".