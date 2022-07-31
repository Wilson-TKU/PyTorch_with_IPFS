# PyTorch_with_IPFS
Use IPFS to download training data and upload the model to IPFS, it is useful in blockchain to save the bandwidth in network transfer.

# Prerequisite

## My Enviroment
- OS: Ubuntu 20.04
- Python: 3.8.11
- IPFS 
    - Golang version: go1.14.4
    - go-ipfs version: 0.7.0
    - Repo version: 10
## Install IPFS and run up.

Here is two way to install IPFS.
- [go-ipfs](https://github.com/ipfs/kubo)
- use snap to install ipfs

    ```bash
    sudo snap install ipfs
    ```

# How to use.
- Run up IPFS server first.

    ```
    ipfs daemon
    ```
    my go-ipfs version: 0.7.0
- Run main project.

    [Download](https://www.kaggle.com/competitions/dogs-vs-cats/data) training data first.

    ```
    python3 train.py
    ```

# Reference
- [Dog_vs_Cat Transfer Learning by Pytorch](https://www.kaggle.com/code/bassbone/dog-vs-cat-transfer-learning-by-pytorch)
- [Pytorch Cats and Dogs Classification](https://www.kaggle.com/code/adinishad/pytorch-cats-and-dogs-classification/notebook)
- [PyTorch - 練習kaggle - Dogs vs. Cats - 使用自定義的 CNN model - HackMD](https://hackmd.io/@lido2370/S1aX6e1nN?type=view)
- [go-ipfs](https://github.com/ipfs/kubo)