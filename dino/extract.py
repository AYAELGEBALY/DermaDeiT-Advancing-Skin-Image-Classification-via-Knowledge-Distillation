import numpy as np
import onnxruntime as ort
import torchvision.transforms as T
from PIL import Image
import os
import json
from tqdm import tqdm
import warnings
import logging
import sys
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from monai.transforms import (
    Compose, Rand2DElastic, RandRotate90, RandFlip, RandAffine,
    RandCoarseShuffle, EnsureType, LoadImage, Resize, ToTensor, NormalizeIntensity,ScaleIntensity
)


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

dino_model_path = "dino/dinov2_vits14.onnx"
ort_session = ort.InferenceSession(dino_model_path, providers=["CPUExecutionProvider"])
print("ONNX model loaded successfully!")
transform_image = T.Compose(
    [
        T.ToTensor(),
        T.Resize(244),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


augmentations = T.Compose([
    ScaleIntensity(0,1),
    Rand2DElastic(prob=0.5, spacing=100, magnitude_range=(3, 12), padding_mode="zeros"),
    RandRotate90(prob=0.5, spatial_axes=[0, 1]),
    RandFlip(prob=0.5, spatial_axis=0),
    RandFlip(prob=0.5, spatial_axis=1),
    RandAffine(prob=0.5,translate_range=(50, 50),rotate_range=(0.75, 0.75),scale_range=(0.1, 0.1),shear_range=(0.1, 0.1),padding_mode="zeros",),
    #RandCoarseShuffle(holes=1, spatial_size=50, max_holes=5, max_spatial_size=150, prob=0.5),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])



def load_image(img: str, train=False) -> np.ndarray:
    img = Image.open(img).convert("RGB")  
    transformed_img = transform_image(img).unsqueeze(0)  
    if train:
        transformed_img = augmentations(transformed_img[0]).unsqueeze(0)
    return transformed_img.numpy()

def compute_embeddings(files: list, output_npy_prefix: str):
    embeddings = []
    with torch.no_grad():
        for file in tqdm(files, desc="Computing embeddings"):
            try:
                input_image = load_image(file, train=True)
                ort_inputs = {ort_session.get_inputs()[0].name: input_image}
                ort_outs = ort_session.run(None, ort_inputs)
                embeddings.append(ort_outs[0].reshape(-1))  
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
    return np.array(embeddings)

def read_melanoma_dataset(dataset_dir):
    image_paths = []
    label_mapping = {}
    LABEL_MAP = {"bcc": 0, "mel": 1, "scc": 2} 
    for label_dir in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_dir)
        if os.path.isdir(label_path):
            label = LABEL_MAP[label_dir]  
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                if os.path.isfile(img_path):
                    image_paths.append(img_path)
                    label_mapping[img_path] = label
    labels = [label_mapping[path] for path in image_paths]  
    return image_paths, np.array(labels)

def main():
    dataset_dir = "/Users/ayaelgebaly/Downloads/maiaUdg/CAD/DLproject/train3C/train"  
    output_npy_prefix = "train_3Cls" 
    print("Reading dataset...")
    files, labels = read_melanoma_dataset(dataset_dir)
    smote = SMOTE()
    labels, indices = smote.fit_resample(labels.reshape(-1, 1), labels)
    files, labels = list(np.array(files)[indices]), labels[:,0]
    print(f"Number of images: {len(files)}")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    for i in range(10):
        embeddings = compute_embeddings(files, output_npy_prefix)
        np.save(f"{output_npy_prefix}_embeddings_{i}.npy", np.array(embeddings))
        print(f"Embeddings saved to {output_npy_prefix}_embeddings_{i}.npy")
        np.save(f"{output_npy_prefix}_labels_{i}.npy", labels)
        print(f"Labels saved to {output_npy_prefix}_labels_{i}.npy")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Labels shape: {labels.shape}")
        print()

if __name__ == "__main__":
    main()
