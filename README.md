# Melanoma Classification with DeiT and Google DermaNet Knowledge Distillation

## Overview
This project implements a hybrid deep learning approach for skin lesion classification by combining DeiT (Data-efficient Image Transformers) with knowledge distillation from Google DermaNet embeddings. The system achieves high accuracy in distinguishing between nevus and other skin conditions. In this task we used some cnn based models to be our basline in this project. 

# by Aya Elgebaly, Muhammed Alberb 

## Architecture
1. Primary Model: DeiT (Data-efficient Image Transformer)
2. Knowledge Distillation: Google DermaNet embeddings
3. Additional Models:
        EfficientNet
        MobileNet
        DINO (Self-supervised Vision Transformer)
        
## Project Structure

├── cnn/
│   ├── dataset.py
│   ├── efficientnet_train.py
│   ├── efficientnet2Cls.ipynb 
│   ├── efficientnet3Cls.ipynb
│   ├── mobilenet_train.py
│   ├── mobilenet_train+aug.py
│   └── swin.py
├── DermaDeiT/
│   ├── DeiT_2Cls.ipynb
│   ├── DeiT_3Cls.ipynb
│   ├── DeiT_and_GoogleDerm_2Cls.ipynb
│   └── DeiT_and_GoogleDerm_3Cls.ipynb
├── dino/
│   ├── classify.py
│   ├── convert_to_onnx.py
│   ├── diffClassifiers.py
│   ├── dlclassifiers.py
│   ├── embaddingExtraction.py
│   ├── extract.py
│   ├── finetuneDino.py
│   ├── optunasvm.py
│   └── sVMs.py
└── requirements.txt

## Features
    Binary classification (nevus vs others)
    Knowledge distillation from Google DermaNet
    Data augmentation pipeline
    Test-time augmentation (TTA)
    Model evaluation metrics
    Support for both 2-class and 3-class classification

## Requirements
    monai==1.4.0
    torch
    albumentations==1.4.21
    numpy==2.1.3
    opencv-python==4.10.0.84
    pandas==2.2.3
    pillow==11.0.0


## Installation
```bash
pip install -r requirements.txt
```

## Model Training
The training process includes:

    Loading pre-trained DeiT model
    Extracting Google DermaNet embeddings
    Knowledge distillation training
    Data augmentation
    Validation monitoring
    Model checkpointing
