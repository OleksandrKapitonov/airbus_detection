"""
Script contains helpful functions.
Like `rle_decode()` (for pixel decoding)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras import losses

# ref https://www.kaggle.com/code/paulorzp/run-length-encode-and-decode/script
def rle_decode(mask_rle):
    '''
    Decodes Encoded pixel sequence
    
    Args:
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        
    Returns:
        A numpy array out of 0's and 1's were 1 is the segment that we want to highlight.
        
        Examples:
            "849759324.jpg", 64 -> "[0,1,0,0,0,1,0,0,0,1,...]"

    '''
    if isinstance(mask_rle, float):
        return np.nan
    shape = (768, 768)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.flatten()


def check_memory() -> None:
    """
    Prints out variables and amount of memory they use

    Returns: Prints out a list of memory usage
    """
    local_vars = list(locals().items())
    for var, obj in local_vars:
        print(var, sys.getsizeof(obj))
    

def mormalize(image: np.array):
    """
    Scales image pixels from 0 to 1
    
    Args:
        image: image as numpy array
        
    Returns:
        Normalized imaged form 0 - 1 scale
        Example:
            [122, 125, 245, 234, ...] -> [0.122, 0,125, 0.245, 0.234, ...]
    """
    return tf.cast(image, tf.float32) / 255.0


def augment(input_image: tf.Tensor,
            input_mask: tf.Tensor):
    """
    Randomly preformce image augmentation such as fliping images
    
    Args:
        input_image: TensorFlow tensor
        input_mask: Coresposding to a tensor mask of TensorFlow tensor type
        
    Returns: Flipped tensor
    """
    if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    return input_image, input_mask


def show_loss(loss_history):
    """
    Plot loss and accuracy curves
    """
    epochs = np.concatenate([mh.epoch for mh in loss_history])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')
    
    _ = ax2.plot(epochs, np.concatenate([mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_binary_accuracy'] for mh in loss_history]), 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('Binary Accuracy (%)')

   
    
def plot_data_balance(df: pd.DataFrame,
                      len_obs: int,
                      miss_labl: int)-> None:
    """
    Plots a bar chart of feature quantities
    
    Args:
        df: Dataframe were currently exploring
        len_obs: Number of all bservations
        miss_labl: Number of mising labels
    
    Returns: None (Creates a barchart to represent variety of features)
    """
    print(f'There are: {len_obs} observations.\n')
    print(f'Missing observations: {miss_labl}\n')
    print(f'The ratio of missed data to all is: {miss_labl/len_obs:.4f}\n')
    print(f'Number of images with ships: {len_obs-miss_labl}\n')

    # Plot the values we get
    data_info = {'data without ships': miss_labl, 'data with a ship': len_obs-miss_labl}
    plt.bar(list(data_info.keys()), list(data_info.values()))
    plt.title('Balance of the dataset.')
    plt.xlabel('type of image')
    plt.ylabel('Quantity')
    plt.show();


def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    """ 
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();    

    
def dice_score(y_true, y_pred, smooth=1):
    """
    Segmentation metric of 2 * the Area of Overlap divided by the total number of pixels in both images
    
    Args:
        y_true: Ground truth value, in our case is a mask of the ship
        y_pred: Predicted value in our case the mask that was predicted
        smooth: Parameter that allows to smooth the output
        
    Returns: 1 - the weighted avarege of the overlaps
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    loss = 1 - score
    loss = losses.binary_crossentropy(y_true, y_pred) + loss
    return loss
    
