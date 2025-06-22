import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, concatenate, Dense, Reshape)
from tensorflow.keras.models import Model


image_height = 512
image_width = 512
n_classes = 2
input_channels = 1
output_channels = n_classes

#Convolutional params
con_layers_1 = 64
con_layers_2 = 128
con_layers_3 = 256
con_layers_4 = 512
con_layers_5 = 1024



#Pooling params
max_pool_size = (2,2)


def jaccard_coeff(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection + 1)

def jaccard_coeff_multiclass(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Compute per-class IoU
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2]) - intersection
    iou = intersection / (union + tf.keras.backend.epsilon())
    return tf.reduce_mean(iou)

def jaccard_loss(y_true, y_pred):
    return -jaccard_coeff(y_true, y_pred)

import tensorflow as tf

def weighted_jaccard_loss(y_true, y_pred, class_weights):
    """
    y_true: one-hot encoded ground truth, shape (batch, H, W, C)
    y_pred: softmax probabilities, shape (batch, H, W, C)
    class_weights: 1D tensor of shape (C,) with weight per class
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)  # avoid log(0) issues

    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])  # per class
    union = tf.reduce_sum(y_true + y_pred - y_true * y_pred, axis=[0, 1, 2])  # per class

    iou = intersection / (union + 1e-7)  # avoid division by zero

    weighted_iou = iou * class_weights
    return 1.0 - tf.reduce_sum(weighted_iou) / tf.reduce_sum(class_weights)

def make_weighted_jaccard_loss(class_weights):
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        return weighted_jaccard_loss(y_true, y_pred, class_weights_tensor)
    return loss


def double_conv(x, out_channels, padding='valid'):
    """(Conv3×3, ReLU) × 2"""
    x = Conv2D(out_channels, kernel_size=3, padding=padding, activation='relu')(x)
    x = Conv2D(out_channels, kernel_size=3, padding=padding, activation='relu')(x)
    return x

def unet_model_same_padding(in_channels = input_channels, out_channels = output_channels):
    inputs = Input(shape=(image_height, image_width, in_channels))

    #---Encoder---
    c1 = double_conv(inputs, con_layers_1, "same")      # -> 512x512x64
    p1 = MaxPooling2D(pool_size = max_pool_size)(c1)    # -> 256x256x64

    c2 = double_conv(p1,con_layers_2, "same")           # -> 256x256x128
    p2 = MaxPooling2D(pool_size = max_pool_size)(c2)    # -> 128x128x128

    c3 = double_conv(p2, con_layers_3, "same")          # -> 128×128×256
    p3 = MaxPooling2D(pool_size = max_pool_size)(c3)    # -> 64×64×256

    c4 = double_conv(p3, con_layers_4, "same")          # -> 64×64×512
    p4 = MaxPooling2D(pool_size = max_pool_size)(c4)    # -> 32×32×512

    c5 = double_conv(p4, con_layers_5, "same")          # Bottleneck: 32×32×1024

    #---decoder---

    u6 = Conv2DTranspose(512, kernel_size=2, strides=2, padding='same')(c5)     # -> 64×64×512
    m6 = concatenate([u6, c4], axis=-1)
    c6 = double_conv(m6, 512, "same")                                           # -> 64×64×512

    u7 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(c6)     # -> 128×128×256
    m7 = concatenate([u7, c3], axis=-1)
    c7 = double_conv(m7, 256, "same")                                           # -> 128×128×256

    u8 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(c7)     # -> 256×256×128
    m8 = concatenate([u8, c2], axis=-1)
    c8 = double_conv(m8, 128, "same")                                           # -> 256×256×128

    u9 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(c8)      # -> 512×512×64
    m9 = concatenate([u9, c1], axis=-1)
    c9 = double_conv(m9, 64, "same")                                            # -> 512×512×64

    outputs = Conv2D(out_channels, kernel_size=1, activation="softmax")(c9)    # -> 512×512×2

    return Model(inputs=inputs, outputs=outputs, name="U-Net_512x512_same_padding")

def unet_metadata_in_bottleneck(in_channels = input_channels, out_channels = output_channels, metadata_channels = 2):
    inputs = Input(shape=(image_height, image_width, in_channels), name="input_image")
    metadata = Input(shape=(metadata_channels,), name="input_metadata")

    #---Encoder---
    c1 = double_conv(inputs, con_layers_1, "same")      # -> 512x512x64
    p1 = MaxPooling2D(pool_size = max_pool_size)(c1)    # -> 256x256x64

    c2 = double_conv(p1, con_layers_2, "same")          # -> 256x256x128
    p2 = MaxPooling2D(pool_size = max_pool_size)(c2)    # -> 128x128x128

    c3 = double_conv(p2, con_layers_3, "same")          # -> 128x128x256
    p3 = MaxPooling2D(pool_size = max_pool_size)(c3)    # -> 64x64x256

    c4 = double_conv(p3, con_layers_4, "same")          # -> 64x64x512
    p4 = MaxPooling2D(pool_size = max_pool_size)(c4)    # -> 32x32x512
    
    # ---bottleneck---
    c5 = double_conv(p4, con_layers_5, "same")          # Bottleneck: 32×32×1024
    
    # Concatenate metadata with the bottleneck feature map for 512x512 input and 2 metadata channels
    x_meta = Dense(512, activation='relu')(metadata)  # Project metadata to higher dimension
    x_meta = Dense(32*32*8, activation='relu')(x_meta)  # Match bottleneck spatial size and some channels
    x_meta = Reshape((32, 32, 8))(x_meta)  # Reshape to (32, 32, 8)

    bottleneck = concatenate([c5, x_meta], axis=-1)  # (32, 32, 1032)

    #---decoder---

    u6 = Conv2DTranspose(512, kernel_size=2, strides=2, padding='same')(bottleneck)     # -> 64×64×512
    m6 = concatenate([u6, c4], axis=-1)
    c6 = double_conv(m6, 512, "same")                                           # -> 64×64×512

    u7 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(c6)     # -> 128×128×256
    m7 = concatenate([u7, c3], axis=-1)
    c7 = double_conv(m7, 256, "same")                                           # -> 128×128×256

    u8 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(c7)     # -> 256×256×128
    m8 = concatenate([u8, c2], axis=-1)
    c8 = double_conv(m8, 128, "same")                                           # -> 256×256×128

    u9 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(c8)      # -> 512×512×64
    m9 = concatenate([u9, c1], axis=-1)
    c9 = double_conv(m9, 64, "same")                                            # -> 512×512×64

    outputs = Conv2D(out_channels, kernel_size=1, activation="softmax")(c9)    # -> 512×512×2

    return Model(inputs=[inputs, metadata], outputs=outputs, name="U-Net_512x512_same_padding")




