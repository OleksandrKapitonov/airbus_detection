"""
Script turns data into tensorflow dataset + image transformations
"""
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import PIL
import tensorflow as tf

train_folder = pathlib.Path(r'./data/train_v2')
test_folder = pathlib.Path(r'./data/test_v2')
save_path = pathlib.Path(r'./data/dataset')

BATCH_SIZE = 32

dataset_df = pd.read_pickle('dataset_df.pkl')

def resize_img(image_name: str) -> np.array:
    """
    Change the size of an image
    
    Args:
        image_name: The name of the image to transform
        mask: Decoded pixel sequence
        
    Returns resized image and mask in a tuple
    """
    image = PIL.Image.open(train_folder / image_name)
    image.thumbnail((128,128))
    
    return np.array(image)


def resize_mask(decode: np.array) -> np.array:
    """
    Change the size of an image
    
    Args:
        image_name: The name of the image to transform
        mask: Decoded pixel sequence
        
    Returns resized image and mask in a tuple
    """
    if decode is 0:
        return np.zeros(shape=(128, 128))
    mask = PIL.Image.fromarray(decode.reshape(768, 768).T)
    mask.thumbnail((128,128))
    
    return np.array(mask)


features = tf.convert_to_tensor(list(dataset_df.ImageId.apply(resize_img)))
labels = tf.convert_to_tensor(list(dataset_df.DecodedPixels.apply(resize_mask)))
labels = tf.expand_dims(labels, axis=-1)
labels = tf.keras.utils.to_categorical(labels, num_classes=3)

# Create a tensorflow dataset
ships_dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(32)

train_threshold = 0.8
train_size = int(len(ships_dataset) * train_threshold)

train_data = ships_dataset.take(train_size)
test_data = ships_dataset.skip(train_size)
