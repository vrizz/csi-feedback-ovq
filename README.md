# User-Driven Adaptive CSI Feedback With Ordered Vector Quantization
[![wandb badge](https://github.com/wandb/assets/blob/main/wandb-github-badge.svg)](https://wandb.ai/valer/csi-feedback-ovq-fine-tune/reports/Adaptive-Channel-Compression-with-Ordered-Vector-Quantization-OVQ---VmlldzoxMzYxMDg2Nw)

🚀 Welcome to the repository of the paper "User-Driven Adaptive CSI Feedback With Ordered Vector Quantization"!  This repository contains the code and resources needed to reproduce the key results from our study. For full details, please refer to the [paper](https://ieeexplore.ieee.org/document/10208156).

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Dataset Preparation
This project uses the **COST2100 "outdoor"** dataset for training and evaluation.

To download and prepare the dataset, run the following commands after cloning the repository:

```
curl -L -o COST2100_dataset.zip https://www.dropbox.com/scl/fo/tqhriijik2p76j7kfp9jl/h?rlkey=4r1zvjpv4lh5h4fpt7lbpus8c&e=2&st=pmf7duk6&dl=1
```

```
unzip COST2100_dataset.zip -d COST2100_dataset
```

```
rm -f COST2100_dataset.zip
```

## Usage

This repository demonstrates the proposed **Ordered Vector Quantization (OVQ)** scheme using **TransNet** and **CRNet** as base models.

### Pretraining

To reproduce the pretraining results, run the following commands:

CRNet

```
python3 main_cost.py -d out -m crnet -r 4 -b 10 -e 4
```

TransNet

```
python3 main_cost.py -d out -m transnet -r 2 -b 10 -e 8
```

### Fine-tuning
To fine-tune the pretrained models and obtain final results, use:

CRNet
```
python3 main_cost.py -d out -m crnet -r 4 -b 10 -e 4 -ft
```

TransNet
```
python3 main_cost.py -d out -m transnet -r 2 -b 10 -e 8 -ft
```

## Results

- The `checkpoints` folder contains the "best_model" checkpoints for both the pretraining and fine-tuning phases.
- The `tables` folder contains `.csv` files that can be used directly for your plots.

✨ You can explore the final results and interactive tables on [Weights & Biases](https://wandb.ai/valer/csi-feedback-ovq-fine-tune/reports/Adaptive-Channel-Compression-with-Ordered-Vector-Quantization-OVQ---VmlldzoxMzYxMDg2Nw).

## Citation
📚 If you find our work helpful in your research, we’d be happy if you cite us!

```bibtex
@article{rizzello2023user,
  author={Rizzello, Valentina and Nerini, Matteo and Joham, Michael and Clerckx, Bruno and Utschick, Wolfgang},
  journal={IEEE Wireless Communications Letters}, 
  title={User-Driven Adaptive CSI Feedback With Ordered Vector Quantization}, 
  year={2023},
  volume={12},
  number={11},
  pages={1956-1960},
  doi={10.1109/LWC.2023.3301992}
}
```

## Acknowledgements

1. **COST2100 dataset**  
   C.-K. Wen, W.-T. Shih, and S. Jin, "Deep Learning for Massive MIMO CSI Feedback,"  
   *IEEE Wireless Communications Letters*, vol. 7, no. 5, pp. 748–751, Oct. 2018.

2. **TransNet base model**  
   Y. Cui, A. Guo, and C. Song, "TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System,"  
   *IEEE Wireless Communications Letters*, vol. 11, no. 5, pp. 903–907, May 2022.

3. **CRNet base model**  
   Z. Lu, J. Wang, and J. Song, "Multi-resolution CSI Feedback with Deep Learning in Massive MIMO System,"  
   *ICC 2020 - IEEE International Conference on Communications*, Dublin, Ireland, 2020, pp. 1–6.

