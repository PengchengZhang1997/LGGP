# Local-Global Graph Pooling via Mutual Information Maximization for Video-Paragraph Retrieval
This code contains PyTorch implementation of the paper "[Local-Global Graph Pooling via Mutual Information Maximization for Video-Paragraph Retrieval](https://ieeexplore.ieee.org/document/9779708)", which has been accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT).
![image](https://user-images.githubusercontent.com/64316571/144455869-215952a2-633d-41b9-b08a-8427762883b8.png)

## Requirements
Python 3.6 and PyTorch 1.6.

Install required packages using the `environment.yml` file.

`conda env create -f environment.yml`

## Datasets
[ActivityNet Captions](https://academictorrents.com/details/0c824440c94cc18ace1cb2c77423919b728d703e), [Youcook2 with ImageNet/Kinetics Features](https://academictorrents.com/details/3ae97c261ed32d3bd5326d3bf6991c9e2ea3dc17), and [Youcook2 with Howto100m features](https://academictorrents.com/details/70417e3793dbbb03ca68981307860254766d5a1d) are used in our experiments.

Here we provide annotations and pretrained features in BaiduNetdisk.

## Training & Inference
```
cd LGGP
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

## Acknownledgements
Our code is based on the implementations of [HGR(CVPR2020)](https://github.com/cshizhe/hgr_v2t) and [COOT(NeurIPS2020)](https://github.com/gingsi/coot-videotext).
