import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, concatenate)
from tensorflow.keras.models import Model


image_height = 322
image_width = 322
n_classes = 4
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

def jaccard_loss_multiclass(y_true, y_pred):
    return -jaccard_coeff_multiclass(y_true, y_pred)

def double_conv(x, out_channels, padding='valid'):
    """(Conv3×3, ReLU) × 2"""
    x = Conv2D(out_channels, kernel_size=3, padding=padding, activation='relu')(x)
    x = Conv2D(out_channels, kernel_size=3, padding=padding, activation='relu')(x)
    return x


def unet_model(in_channels = input_channels, out_channels = output_channels):
    inputs = Input(shape=(image_height, image_width, in_channels))

    #---Encoder---
    c1 = double_conv(inputs, con_layers_1)                  # -> 318x318x64
    p1 = MaxPooling2D(pool_size = max_pool_size)(c1)        # -> 159x159x64

    c2 = double_conv(p1,con_layers_2)                       # -> 155x155x128
    p2 = MaxPooling2D(pool_size = max_pool_size)(c2)        # -> 77x77x128

    c3 = double_conv(p2, con_layers_3)                      # -> 73×73×256
    p3 = MaxPooling2D(pool_size = max_pool_size)(c3)        # -> 36×36×256

    c4 = double_conv(p3, con_layers_4)                      # -> 32×32×512
    p4 = MaxPooling2D(pool_size = max_pool_size)(c4)        # -> 16×16×512

    c5 = double_conv(p4, con_layers_5)                      # Bottleneck: 12×12×1024

    #---decoder---

    u6 = Conv2DTranspose(512, kernel_size=2, strides=2, padding='valid')(c5)    # -> 26×26×512
    # crop c4 (32×32) → (26×26): Crop 3 Top/Left, 3 Bottom/Right (32-26=6)
    c4_crop = Cropping2D(cropping=((3,3),(3,3)))(c4)
    m6   = concatenate([u6, c4_crop], axis=-1)
    c6   = double_conv(m6, 512)

    u7 = Conv2DTranspose(256, 2, strides=2)(c6)                                 # -> 54×54×256
    # crop c3 (73×73) → (54×54): Crop 9/10 each side (73-54=19 → 9 top,10 bottom)
    c3_crop = Cropping2D(cropping=((9,10),(9,10)))(c3)
    m7   = concatenate([u7, c3_crop], axis=-1)
    c7   = double_conv(m7, 256)

    u8 = Conv2DTranspose(128, 2, strides=2)(c7)                                 # -> 110×110×128
    # crop c2 (155×155) → (110×110): Crop 22/23 (155-110=45 → 22,23)
    c2_crop = Cropping2D(cropping=((22,23),(22,23)))(c2)
    m8   = concatenate([u8, c2_crop], axis=-1)
    c8   = double_conv(m8, 128)

    u9 = Conv2DTranspose(64, 2, strides=2)(c8)                                  # -> 222×222×64
    # crop c1 (318×318) → (222×222): Crop 48 each side ((318-222)=96 → 48,48)
    c1_crop = Cropping2D(cropping=((48,48),(48,48)))(c1)
    m9   = concatenate([u9, c1_crop], axis=-1)
    c9   = double_conv(m9, 64)

    # --- Ausgabe 1×1 Conv ---

    outputs = Conv2D(out_channels, kernel_size=1, activation=None)(c9)
    
    return Model(inputs=inputs, outputs=outputs, name="U-Net_322x322")

def unet_model_same_padding(in_channels = input_channels, out_channels = output_channels):
    inputs = Input(shape=(image_height, image_width, in_channels))

    #---Encoder---
    c1 = double_conv(inputs, con_layers_1, "same")      # -> 322x322x64
    p1 = MaxPooling2D(pool_size = max_pool_size)(c1)    # -> 161x161x64

    c2 = double_conv(p1,con_layers_2, "same")           # -> 161x161x128
    p2 = MaxPooling2D(pool_size = max_pool_size)(c2)    # -> 80x80x128

    c3 = double_conv(p2, con_layers_3, "same")          # -> 80×80×256
    p3 = MaxPooling2D(pool_size = max_pool_size)(c3)    # -> 40×40×256

    c4 = double_conv(p3, con_layers_4, "same")          # -> 40×40×512
    p4 = MaxPooling2D(pool_size = max_pool_size)(c4)    # -> 20×20×512

    c5 = double_conv(p4, con_layers_5, "same")          # Bottleneck: 20×20×1024

    #---decoder---

    u6 = Conv2DTranspose(512, kernel_size=2, strides=2, padding='same')(c5)     # -> 40×40×512
    m6   = concatenate([u6, c4], axis=-1)
    c6   = double_conv(m6, 512, "same")                                         # -> 40×40×512

    u7 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(c6)     # -> 80×80×256
    m7   = concatenate([u7, c3], axis=-1)
    c7   = double_conv(m7, 256, "same")                                         # -> 80×80×256

    u8 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(c7)     # -> 160×160×128
    u8 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(u8)            # -> 161×161×128
    m8   = concatenate([u8, c2], axis=-1)
    c8   = double_conv(m8, 128, "same")                                         # -> 161×161×128

    u9 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(c8)      # -> 322×322×64
    m9   = concatenate([u9, c1], axis=-1)
    c9   = double_conv(m9, 64, "same")                                          # -> 322×322×64

    # --- Ausgabe 1×1 Conv ---
    outputs = Conv2D(out_channels, kernel_size=1, activation="softmax")(c9)     # -> 322×322×4
    
    return Model(inputs=inputs, outputs=outputs, name="U-Net_322x322_same_padding")




