import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, concatenate)
from tensorflow.keras.models import Model


image_height = 322
image_width = 322
input_channels = 1
output_channels = 2

#Convolutional params
con_layers_1 = 64
con_layers_2 = 128
con_layers_3 = 256
con_layers_4 = 512
con_layers_5 = 1024



#Pooling params
max_pool_size = 2

def double_conv(x, out_channels):
    """(Conv3×3, ReLU) × 2, valid padding"""
    x = Conv2D(out_channels, kernel_size=3, padding='valid', activation='relu')(x)
    x = Conv2D(out_channels, kernel_size=3, padding='valid', activation='relu')(x)
    return x


def unet_model(in_channels = input_channels, out_channels = output_channels):
    inputs = Input(shape=(image_height, image_width, input_channels))

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




