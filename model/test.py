import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("GPUs found:", gpus)

if gpus:
    print("TensorFlow is using GPU")
else:
    print("No GPU detected by TensorFlow")

print("TensorFlow version:", tf.__version__)

build_info = tf.sysconfig.get_build_info()
print("TF CUDA version:", build_info.get("cuda_version", "N/A"))
print("TF cuDNN version:", build_info.get("cudnn_version", "N/A"))

