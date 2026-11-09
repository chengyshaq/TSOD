# TSOD
official PyTorch implementation for paper: 
"Long‑tail Out‑of‑Distribution Detection via Distribution Learning and Dual‑metric Triplet Optimization"

# TSOD proposes a collaborative framework 
consisting of energy discrimination, distribution alignment and feature matching, 
together with a dual‑metric triplet constraint module for long‑tail out‑of‑distribution detection.
This repo only supports CIFAR10‑LT and CIFAR100‑LT datasets.  
Long‑tail imbalance ratio ρ = 100 by default.

# Dependencies
- Python 3.9.13
- Pytorch 1.12.1
- Torchvision 0.13.1
- Numpy 1.23.1
- CUDA 11.5 
- PIL 9.5.0
- scikit-learn 1.2.2

# How to get the dataset
The datasets used and/or analysed during the current study are available in the following repositories/official websites with their respective web links: 
(1) CIFAR
You can get the CIFAR10 and CIFAR100 training and test dataset from the official website:https://www.cs.toronto.edu/~kriz/cifar.html. 
(2) 300K random images
You can get the 300k random images dataset from the GitHub repository: https://github.com/hendrycks/outlier-exposure. 
Alternatively, you can get this dataset directly from the following url: https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy.
(3) SCOOD
You can get the SCOOD datasets from the GitHub repository: https://github.com/Jingkang50/ICCV21_SCOOD. Alternatively, you can get the datasets in the project page: https://jingkang50.github.io/projects/scood.

# Alternatively, download the experimental dataset using the following method
# CIFAR 
You can get the CIFAR10 and CIFAR100 training and test dataset from the official website: https://www.cs.toronto.edu/~kriz/cifar.html.
- CIFAR10<br>
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
- CIFAR100<br>
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```
Training set:
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```
Test set: 
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
```

# How to run the code
# Train
- CIFAR10-LT <br>
```bash
python train.py --ds cifar10 --drp <where_you_store_all_your_datasets> --suffix exp0 --gpu 0
```
- CIFAR100-LT <br>
```bash
python train.py --ds cifar100 --drp <where_you_store_all_your_datasets> --suffix exp0 --gpu 0
```

You can experiment on various settings of "Lambda" or "imbalance_ratio" by running the following as an example:<br>
```bash
python train.py --ds cifar10 --suffix Lambda0.6 --Lambda 0.6 --gpu 0
python train.py --ds cifar100 --suffix rho0.1 --imbalance_ratio 0.1 --gpu 0
```

# Test
- CIFAR10-LT<br>
```bash
for dout in texture svhn cifar tin lsun places365
do
python test.py --gpu 0 --ds cifar10 --dout $dout 
    --drp <where_you_store_all_your_datasets> 
    --ckpt_path <where_you_save_the_ckpt>
done
```
- CIFAR100-LT<br>
```bash
for dout in texture svhn cifar tin lsun places365
do
python test.py --gpu 0 --ds cifar100 --dout $dout 
    --drp <where_you_store_all_your_datasets> 
    --ckpt_path <where_you_save_the_ckpt> 
done
```

# OOD test datasets (SC‑OOD benchmark)
We evaluate OOD detection performance on SC‑OOD benchmark:
- Far‑OOD test sets: `Texture`, `SVHN`, `LSUN`, `TinyImageNet`, `Places365`
- Near‑OOD setting:
  - For CIFAR10‑LT: take CIFAR‑100 as near‑OOD test set
  - For CIFAR100‑LT: take CIFAR‑10 as near‑OOD test set

The test code will print out the evaluation results of 3 OOD performance metrics (AUROC, AUPR, FPR95) 
and classification performance metric (ACC).

# Repository Structure
SOD/
├── data/
│   ├── cifar_lt.py          # build long‑tail CIFAR10‑LT / CIFAR100‑LT
│   ├── scood_dataloader.py  # dataloader for SC‑OOD evaluation
│   └── tinyimages.py        # load auxiliary TinyImages OOD samples
├── models/
│   ├── backbone.py          # ResNet18 feature extractor
│   ├── loss_ood.py          # energy loss, KL divergence, multi‑kernel MMD loss
│   └── loss_dual_triplet.py # dual‑metric triplet loss (Euclidean + angular distance)
├── train.py                 # main training entry
├── test.py                  # evaluation script
├── utils.py                 # metrics( AUROC / AUPR / FPR95 / ACC ), logging, clamp operation for arccos
└── README.md

# Evaluation
# Evaluate all OOD test sets for CIFAR10‑LT:
for dout in texture svhn cifar100 tinyimagenet lsun places365
do
python test.py \
  --gpu 0 \
  --ds cifar10-lt \
  --dout $dout \
  --drp ./data \
  --ckpt_path ./checkpoints/exp_cifar10lt/best.pt
done
# Evaluate all OOD test sets for CIFAR100‑LT:
for dout in texture svhn cifar10 tinyimagenet lsun places365
do
python test.py \
  --gpu 0 \
  --ds cifar100-lt \
  --dout $dout \
  --drp ./data \
  --ckpt_path ./checkpoints/exp_cifar100lt/best.pt
done

# Initial parameter values for experimental training
# on the CIFAR-10 dataset, 
the total loss weights were set to λ1 = 0.4 and λ2 = 0.6, 
the dynamic threshold for energy loss was τ = 1.0, 
the dynamic fusion coefficient for KL divergence was γ = 0.2, 
in the dual-metric triplet loss, the Euclidean distance coefficient η1 = 0.7, 
the angular distance coefficient η2 = 0.3, and the boundary margin m = 1.0;
# on the CIFAR-100 dataset, the total loss weights were set to λ1 = 0.15 and λ2 = 0.4, 
the dynamic threshold for energy loss was τ = 1.0, 
the dynamic fusion coefficient for KL divergence was γ = 0.6,
in the dual-metric triplet loss, the Euclidean distance coefficient η1 = 0.5, 
the angular distance coefficient η2 = 0.5, and the boundary margin m = 1.0.

# The packages that must be imported for the experimental running environment
import argparse
import torch
import torch.nn.functional as F
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
import torchvision.transforms as transforms
from torchvision import datasets
from datasets.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.ImbalanceImageNet import LT_Dataset
from datasets.tinyimages_300k import TinyImages
from models.resnet import ResNet18
from models.resnet_imagenet import ResNet50
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.nn.utils import spectral_norm
from collections import Counter

# Code that needs to be debugged due to environmental updates
# 训练主循环
    for epoch in range(args.epochs):
        if args.ddp:
            train_sampler.set_epoch(epoch)

        model.train()

        training_loss_sum = 0.0
        training_loss_LA_sum = 0.0
        training_loss_triplet_sum = 0.0
        training_loss_ood_sum = 0.0
        batch_count = 0

        current_lr = scheduler.get_last_lr()[0]

        # ------------------- 损失权重动态调度（对齐旧成功版本思路） -------------------
        # 三元组延后启动，改用linear慢速上升，降低峰值
        triplet_start_epoch = int(args.epochs * 0.4)
        max_triplet_epochs = max(MIN_DYNAMIC_EPOCHS, int(args.epochs * 0.25))
        if epoch < triplet_start_epoch:
            triplet_weight = 0.0
        elif epoch < triplet_start_epoch + max_triplet_epochs:
            triplet_weight = compute_dynamic_weight(
                epoch - triplet_start_epoch, max_triplet_epochs,
                initial_weight=0.0, final_weight=0.4,
                curve='linear'  # 关键：从 exponential → linear，避免权重突然冲上去
            )
        else:
            triplet_weight = 0.4

        # OOD保持原设置不变
        ood_start_epoch = int(args.epochs * 0.5)
        max_ood_epochs = max(MIN_DYNAMIC_EPOCHS, int(args.epochs * 0.3))
        if epoch < ood_start_epoch:
            ood_weight = 0.0
        elif epoch < ood_start_epoch + max_ood_epochs:
            ood_weight = compute_dynamic_weight(
                epoch - ood_start_epoch, max_ood_epochs,
                initial_weight=0.0, final_weight=MAX_OOD_WEIGHT,
                curve='linear'
            )
        else:
            ood_weight = MAX_OOD_WEIGHT

        for batch_idx, ((in_data, labels), (ood_data, _)) in enumerate(zip(train_loader, ood_loader)):
            in_data, labels = in_data.to(device), labels.to(device)
            ood_data = ood_data.to(device)

            N_in = len(labels)
            all_data = torch.cat([in_data, ood_data], dim=0)

            all_logits, all_reps = model(all_data)
            in_logits = all_logits[:N_in]
            in_reps = all_reps[:N_in]
            ood_logits = all_logits[N_in:]

            adjusted_in_logits = in_logits + args.tau * prior.log()[None, :]
            LA_loss = F.cross_entropy(adjusted_in_logits, labels)

            # ========== 重点改动：margin固定，不再随epoch放大 ==========
            if len(in_reps) > 1 and len(torch.unique(labels)) > 1:
                margin = args.margin
                triplet_loss = batch_hard_triplet_loss(
                    in_reps, labels, logits=in_logits, margin=margin
                )
            else:
                triplet_loss = torch.tensor(0.0, device=device)

            ood_loss = ood_detection_loss_improved(ood_logits, prior, temperature=args.temperature)

            loss = LA_loss + triplet_weight * triplet_loss + ood_weight * ood_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss_sum += loss.item()
            training_loss_LA_sum += LA_loss.item()
            training_loss_triplet_sum += triplet_loss.item()
            training_loss_ood_sum += ood_loss.item()
            batch_count += 1

            if batch_idx % 100 == 0:
                train_str = 'epoch %d batch %d (train): ' \
                            'loss %.4f (LA: %.4f, Triplet: %.4f, OOD: %.4f) | ' \
                            'lr %.6f, triplet_w: %.4f, ood_w: %.4f' % (
                                epoch, batch_idx,
                                training_loss_sum / batch_count,
                                training_loss_LA_sum / batch_count,
                                training_loss_triplet_sum / batch_count,
                                training_loss_ood_sum / batch_count,
                                current_lr, triplet_weight, ood_weight
                            )
                print(train_str)
                if fp:
                    fp.write(train_str + '\n')
                    fp.flush()

        scheduler.step()

        epoch_avg_loss = training_loss_sum / batch_count
        epoch_avg_LA_loss = training_loss_LA_sum / batch_count
        epoch_avg_triplet_loss = training_loss_triplet_sum / batch_count
        epoch_avg_ood_loss = training_loss_ood_sum / batch_count
        epoch_str = f'Epoch {epoch} completed: Avg Loss = {epoch_avg_loss:.4f}, ' \
                    f'LA Loss = {epoch_avg_LA_loss:.4f}, ' \
                    f'Triplet Loss = {epoch_avg_triplet_loss:.4f}, ' \
                    f'OOD Loss = {epoch_avg_ood_loss:.4f}'
        print(epoch_str)
