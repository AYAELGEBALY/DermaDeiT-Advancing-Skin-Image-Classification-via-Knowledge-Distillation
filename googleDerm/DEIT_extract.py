import os
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import from_pretrained_keras


# loaded_model = from_pretrained_keras("google/derm-foundation")
# infer = loaded_model.signatures["serving_default"]

MODEL_PATH = "/Users/ayaelgebaly/.cache/huggingface/hub/models--google--derm-foundation/snapshots/c6c4db06d78456eec54449221950fa76fa1f58be"
print("Loading model using TFSMLayer...")
loaded_model = keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

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
        #output = infer(inputs=tf.constant([input_tensor]))
        output = loaded_model(tf.constant([input_tensor]))
        embedding = output["embedding"].numpy().flatten()
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


cls_to_idx = {'nevus':0, 'others':1}
idx_to_cls = {0:'nevus', 1:'others'}

def get_data(data_path):
  data = []
  classes = [f for f in os.listdir(data_path) if not f.startswith('.')]
  for cls in classes:
      cls_path = os.path.join(data_path, cls)
      for image_name in os.listdir(cls_path):
          image_path = os.path.join(cls_path, image_name)
          data.append({'image':image_path, 'label':cls_to_idx[cls]})
  return data


def save_to_npz(paths, features, labels, file_name):
    np.savez(file_name, paths=paths, features=features, labels=labels)

def load_from_npz(file_name):
    data = np.load(file_name, allow_pickle=True)
    paths = data['paths']
    features = data['features']
    labels = data['labels']
    return paths, features, labels

def extract_embeddings_from_dataset(dataset_dir, output_prefix):
    embeddings = []
    paths = []
    labels = []
    data=get_data(dataset_dir)
    for item in tqdm(data):
        image_path, label = item['image'], item['label']
        embedding = extract_embedding(image_path)
        if embedding is not None:
            embeddings.append(embedding)
            labels.append(label)
            paths.append(image_path)
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    paths = np.array(paths, dtype=object)
    save_to_npz(paths, embeddings, labels, f"{output_prefix}.npz")
    return embeddings, labels


def main():
    dataset_dir = "/Users/ayaelgebaly/Downloads/maiaUdg/CAD/DLproject/train2C/train"
    output_prefix = "google_derm_train2c_embeddings"  
    print(f"Extracting embeddings from dataset: {dataset_dir}")
    embeddings, labels = extract_embeddings_from_dataset(dataset_dir, output_prefix)
    print(f"Extracted {embeddings.shape[0]} embeddings with shape {embeddings.shape[1]}")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
if __name__ == "__main__":
    main()
