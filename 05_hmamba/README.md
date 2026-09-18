# HMamba

<p align="left"><img src="https://github.com/Fuann/hmamba/blob/master/pics/hmamba.png?raw=true" alt="hmamba" width="800"/></p>

## Introduction

This repository contains the official implementation of the paper, [Towards Efficient and Multifaceted Computer-assisted Pronunciation Training Leveraging Hierarchical Selective State Space Model and Decoupled Cross-entropy Loss](https://aclanthology.org/2025.naacl-long.98) (NAACL 2025).

> Codes are based on the open-source repository, [GOPT (Gong et. al, ICASSP 2022)](https://github.com/YuanGongND/gopt).

## Citation

If you find this repository useful, please cite the following paper:

``` bibtex
@inproceedings{chao-chen-2025-towards,
    title = "Towards Efficient and Multifaceted Computer-assisted Pronunciation Training Leveraging Hierarchical Selective State Space Model and Decoupled Cross-entropy Loss",
    author = "Chao, Fu-An  and Chen, Berlin",
    editor = "Chiruzzo, Luis  and Ritter, Alan  and Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.98/",
    pages = "1947--1961",
    ISBN = "979-8-89176-189-6"
```

## Usage

- **Step 1 - Prepare features**

  Please download the files via this [dropbox link](https://www.dropbox.com/scl/fi/2avige1colrltska5746i/data.zip?rlkey=hr5ahgyiihvshnx4f6fm5z7i6&st=iemoz1ot&dl=1) and unzip them into this directory.

- **Step 2 - HMamba**

  A. Compile your kaldi
  ``` yaml
  git clone https://github.com/kaldi-asr/kaldi <your-kaldi-path>
  cd <your-kaldi-path>
  # recommanded version
  git reset --hard d6198906fbb0e3cfa5fae313c7126a78d8321801
  # compile kaldi under tools/ and src/ (see INSTALL for details)
  ```
  
  B. Environment
  ``` yaml
  # create conda environment for CUDA 11.8
  conda create -n hmamba python==3.10.13
  conda activate hmamba
  conda install cudatoolkit==11.8 -c nvidia
  conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
  conda install packaging
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

  # use 'which nvcc' to locate CUDA path
  export CUDA_HOME=/path/to/miniconda3/envs/hmamba

  # install requirements
  pip install -r requirements.txt

  # modify your 'KALDI_ROOT' and 'conda path' in path.sh.
  nano/vim path.sh
  ```

  C. Experiment
  ``` yaml
  bash run.sh
  ```
