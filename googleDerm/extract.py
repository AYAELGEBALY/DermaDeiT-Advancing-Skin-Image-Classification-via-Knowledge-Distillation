import os
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "/Users/ayaelgebaly/.cache/huggingface/hub/models--google--derm-foundation/snapshots/c6c4db06d78456eec54449221950fa76fa1f58be"
print("Loading model using TFSMLayer...")
loaded_model = keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
print("Model loaded successfully!")

def preprocess_image(image_path):
    img = Image.open(image_path)
    buf = BytesIO()
    img.convert("RGB").save(buf, "PNG")
    image_bytes = buf.getvalue()
    input_tensor = tf.train.Example(features=tf.train.Features(
        feature={
            "image/encoded": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_bytes])
            )
        }
    )).SerializeToString()
    return input_tensor

def extract_embedding(image_path):
    try:
        input_tensor = preprocess_image(image_path)
        output = loaded_model(tf.constant([input_tensor]))
        embedding = output["embedding"].numpy().flatten()
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_embeddings_from_dataset(dataset_dir, output_prefix):
    embeddings = []
    labels = []
    label_map = {"nevus": 0, "others": 1} 
    for label_name, label_id in label_map.items():
        class_dir = os.path.join(dataset_dir, label_name)
        if os.path.isdir(class_dir):
            for image_file in tqdm(os.listdir(class_dir), desc=f"Processing {label_name}"):
                image_path = os.path.join(class_dir, image_file)
                embedding = extract_embedding(image_path)
                if embedding is not None:
                    embeddings.append(embedding)
                    labels.append(label_id)
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    np.save(f"{output_prefix}_embeddings.npy", embeddings)
    np.save(f"{output_prefix}_labels.npy", labels)
    print(f"Saved embeddings to {output_prefix}_embeddings.npy")
    print(f"Saved labels to {output_prefix}_labels.npy")
    return embeddings, labels

def main():
    dataset_dir = "/Users/ayaelgebaly/Downloads/maiaUdg/CAD/DLproject/val2C/val"
    output_prefix = "derm_val"  
    print(f"Extracting embeddings from dataset: {dataset_dir}")
    embeddings, labels = extract_embeddings_from_dataset(dataset_dir, output_prefix)
    print(f"Extracted {embeddings.shape[0]} embeddings with shape {embeddings.shape[1]}")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
if __name__ == "__main__":
    main()
