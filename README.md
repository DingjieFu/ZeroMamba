<div align="center">
<h1> ZeroMamba </h1>
</div>

## Notes
This repository includes the following materials for testing and checking our results reported in our paper:

1. **<font color=blue>The testing codes</font>**
2. **<font color=blue>The trained models</font>**

## Results
Results of our released models using various evaluation protocols on three datasets, both in the CZSL and GZSL settings.

| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-------: | :-----: | :-----: | :-----: |
|   CUB   |   80.0    |  72.1   |  76.4   |  74.2   |
|   SUN   |   72.4    |  56.5   |  41.4   |  47.7   |
|  AWA2   |   71.9    |  67.9   |  87.6   |  76.5   |


## Environments setup 

**Note: We highly recommend that you adhere to the following steps.**

- **Python & PyTorch**
  
  ```shell
  conda create -n zeromamba python=3.10.13
  conda activate zeromamba
  conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch- cuda=11.8 -c pytorch -c nvidia
  ```
  
- **Mamba dependencies**

  - download [mamba_ssm-1.1.1](https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
    - `pip install <your file path> `

  - download [casual_conv1d-1.1.0](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.0/causal_conv1d-1.1.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
    - `pip install <your file path> `

- **Vision Mamba dependencies**

  - `cp -r ZeroMamba/VisionMambaModels/Vim/mamba_ssm <your env's site-packages path>`

  - `cd ZeroMamba/VisionMambaModels/VMamba/kernels/selective_scan && pip install .`

## Testing

### Preparing Dataset and Model

We provide trained models and extracted features ([Google Drive]())on three different datasets: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), [AWA2](http://cvml.ist.ac.at/AwA2/) in the CZSL and GZSL settings. The structure:
```
ZeroMamba/
├── data
│   ├── attribute
│   ├── dataset
│   │   ├── AWA2
│   │   │   ├── Animals_with_Attributes2
│   │   │   └── ...
│   │   ├── CUB
│   │   │   ├── CUB_200_2011
│   │   │   └── ...
│   │   ├── SUN
│   │   │   ├── images
│   │   │   └── ...
│   │   ├── xlsa
│   │   └── ...
│   ├── w2v
│   └── ...
├── extract_feature
└── ...
```
