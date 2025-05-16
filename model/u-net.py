import tensorflow as tf
from tensorflow.keras import layers, Model

image_height = 322
image_width = 322
image_channels = 1

def unet_model(input_size=(image_height, image_width, image_channels)):
    inputs = tf.keras.Input(shape = input_size)

    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)



