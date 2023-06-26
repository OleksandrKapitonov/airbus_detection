
"""
This python script contains a unet architecture neural network  
"""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import models, layers

def double_conv_block(x: tf.Tensor,
                      n_filters: int):
    """
    Passes inputs through 2 conv2d and ReLU activation layers
    Conv block -> Conv block
    
    Args:
        x: TensorFlow tensor we pass through
        n_filters: number of filters we want conv layer to use
        
    Returns: A tensor that was modified by the layers in this function
    """
    x = layers.Conv2D(n_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(n_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)

    return x


def downsample_block(x: tf.Tensor,
                     n_filters:int):
    """
    Block that will perform a downsampling operation
    Double_Conv block -> MaxPool(window size 2) -> Dropout(probability 0.3)
    
    Args:
        x: TensorFlow tensor we pass through
        n_filters: number of filters we want conv layer to use
        
    Returns: A tensor that was modified by the layers in this function
    """
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    
    return f, p

def upsample_block(x: tf.Tensor,
                   conv_features: tf.Tensor,
                   n_filters:int):
    """
    Block that will perform a upsampling operation
    Conv Transpose (Unconv) block -> concatenate -> Dropout -> Double_Conv
    
    Args:
        x: TensorFlow tensor we pass through
        conv_features: Tensor which will be concatenated with the one we passed as x
        n_filters: number of filters we want conv layer to use
    
    Returns: A tensor that was modified by the layers in this function
    """
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding='same')(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    
    return x
    
def build_unet_model(input_shape: Tuple[int, int, int]):
    """
    Returns a U-net architecture neural network model
    
    Args:
        x: TensorFlow tensor we pass through
        conv_features: Tensor which will be concatenated with the one we passed as x
        n_filters: number of filters we want conv layer to use
   
    Returns: A U-net architecture model
    """
    # inputs
    inputs = layers.Input(shape=input_shape)

    # encoder
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)
    
    # bottleneck
    bottleneck = double_conv_block(p4, 1024)
    
    # decoder
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)
    
    # outputs
    outputs = layers.Conv2D(3, 1, padding='same', activation='softmax')(u9)
    
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name='U-net')
        
    return unet_model
