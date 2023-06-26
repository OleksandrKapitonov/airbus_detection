"""
File makes inference on the test folder by given image name
"""
import pathlib
import PIL
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from helper_functions import dice_score, plot_loss_curves
from train_u_net import save_path, model_0

test_folder = pathlib.Path(r'./data/test_v2')

def resize_img(image_name: str) -> np.array:
    """
    Change the size of an image
    
    Args:
        image_name: The name of the image to transform
        mask: Decoded pixel sequence
        
    Returns resized image and mask in a tuple
    """
    image = PIL.Image.open(image_name)
    image.thumbnail((128,128))
    
    return tf.convert_to_tensor(image)

inference_path = input('Enter your image path ') + '.jpg'
test_image = resize_img(test_folder / inference_path)

plt.imshow(model_0.predict(tf.expand_dims(test_image, axis=0)).reshape(128, 128, 3))
