import u_net
import os
import cv2
import numpy as np
import tensorflow as tf
import random
print(tf.config.list_physical_devices('GPU'))

X_TRAIN_PATH = 'datasets/Dataset_UHCSDB/Patched/images_patched/'
Y_TRAIN_PATH = 'datasets/Dataset_UHCSDB/Patched/labels_png_patched/'

train_images = []
train_masks = []
n_classes = 4

# Load the training data
for file in os.listdir(X_TRAIN_PATH):
    if file.endswith('.png'):
        img = cv2.imread(os.path.join(X_TRAIN_PATH, file), cv2.IMREAD_GRAYSCALE)
        train_images.append(img)

for file in os.listdir(Y_TRAIN_PATH):
    if file.endswith('.png'):
        img = cv2.imread(os.path.join(Y_TRAIN_PATH, file), cv2.IMREAD_GRAYSCALE)
        train_masks.append(img)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)

# Turn the lists into numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

print("Class values in Y_train:")
unique_classes = np.unique(Y_train)
print(unique_classes)   # [0 1 2 3]

# print shape of X_train and Y_train
print("Shape of X_train:", X_train.shape)  # (number_of_images, height, width, channels)
print("Shape of Y_train:", Y_train.shape)  # (number_of_images, height, width, channels)
# both are (76, 322, 322)

# Convert Y_train to categorical
from keras.utils import to_categorical
train_masks_cat = to_categorical(Y_train, num_classes=n_classes)
Y_train_cat = train_masks_cat.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], n_classes))

test_masks_cat = to_categorical(Y_test, num_classes=n_classes)
Y_test_cat = test_masks_cat.reshape((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], n_classes))

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(Y_train),
    y=Y_train.flatten()
)
print("Class weights:", class_weights)  # [0.338 1.862 2.514 9.083]

model = u_net.unet_model_same_padding()
model.compile(optimizer='adam', loss=[u_net.jaccard_loss_multiclass], metrics=[u_net.jaccard_coeff_multiclass])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

history = model.fit(
    X_train,
    Y_train_cat,
    batch_size=8,
    verbose=1,
    epochs=50,
    validation_data=(X_test, Y_test_cat),
    shuffle=True,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)

model.save('unet_model.keras')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test_cat)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# plot the training history
import matplotlib.pyplot as plt
def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['jaccard_coeff_multiclass'])
    plt.plot(history.history['val_jaccard_coeff_multiclass'])
    plt.title('Model IoU')
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

#plot_history(history)

# Pick a random test image
idx = random.randint(0, X_test.shape[0] - 1)
test_img = X_test[idx]
true_mask = Y_test[idx]

# Predict the mask
pred_mask = model.predict(np.expand_dims(test_img, axis=0))
pred_mask = np.argmax(pred_mask, axis=-1)[0]

# Plot the images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Test Image")
plt.imshow(test_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("True Mask")
plt.imshow(true_mask, cmap='nipy_spectral', vmin=0, vmax=n_classes-1)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=n_classes-1)
plt.axis('off')

plt.show()
