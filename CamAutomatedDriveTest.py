import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts

import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import load_model

import pandas as pd
import ntpath

import matplotlib.image as mpimg
import cv2
import random

from imgaug import augmenters as iaa

datadir = 'Data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns)
pd.set_option('display.max_colwidth', -1)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

print(data.head())

# which steering angle - plot
num_bins = 25

# make data more uniform by setting threshold with samples_per_bin
samples_per_bin = 300

hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+bins[1:])/2
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])),
        (samples_per_bin, samples_per_bin))
# plt.show()
plt.close()

remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        # if data belongs to the same bin
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            # store the number of the item from the common list to the bin's list
            list_.append(i)
    # and shuffle it
    list_ = shuffle(list_)
    # and trim the list of indices to be removed
    list_ = list_[samples_per_bin:]
    # add it to the new common list
    remove_list.extend(list_)

print('removed:', len(remove_list))
# remove the data lines with indices from 'remove_list'
data.drop(data.index[remove_list], inplace=True)
print('remaining:', len(data))

hist, _ = np.histogram(data['steering'], num_bins)

plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])),
        (samples_per_bin, samples_per_bin))
# plt.show()
plt.close()


def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        # use strip to remove all spaces in the path (there are none but..)
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

# both array indices correspond to each other
image_paths, steerings = load_img_steering(datadir+'/IMG', data)

X_train, X_valid, y_train, y_valid = tts(image_paths, steerings, test_size=0.2, random_state=5)

print('Training samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Training set')
axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
axes[1].set_title('Validation set')

# plt.show(fig)
plt.close(fig)

# augmentation with zoom in
def zoom(image):
    # augmented zoom in limit to 30%
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image

# augmentation with pan in x an y directions
def pan(image):
    # augmented pan by 10% in both directions
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

# augmentation with brightness regulation
def brightness(image):
    # augmentation with brightness level between 20%..120%
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

# augmentation with image flipping (mirroring)
# not a part of imgaug library - use opencv instead.
# The steering_angle shall be inverted together
# Remark: essentially it allows for better data balancing
def flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle =- steering_angle
    return image, steering_angle

# add preprocessing to images
def img_preprocess(img):
    img = mpimg.imread(img)
    # crop scenery and bonnet from images
    # remove everything in 0..60 and in 135..140
    img = img[60:135, :, :]
    # change color space. Use YUV format instead of RGB (Recommended by nVIDIA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # resize
    img = cv2.resize(img, (200, 66))
    # normalize
    img = img/255
    return img

# apply augmentations randomly
def random_augmentation(image, steering_angle):
    # image = mpimg.imread(image)
    # rand generates a number from 0..1
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = flip(image, steering_angle)
    return image, steering_angle


# apply preprocessing to some image just for checking
image = image_paths[100]
steering = steerings[100]

# 10 - rows, 2 - columns
fig_img, axs = plt.subplots(10, 2, figsize=(15, 50))
fig_img.tight_layout()

for i in range(10):
    random_i = random.randint(0, len(image_paths) - 1)
    img = mpimg.imread(image_paths[random_i])
    steer = steerings[random_i]
    augmented_image, new_steering = random_augmentation(img, steer)

    axs[i][0].imshow(img)
    axs[i][0].set_title('Original img')
    axs[i][1].imshow(augmented_image)
    axs[i][1].set_title('Augm img')
# plt.show(fig_img)

# original_image = random_augmentation(image, steerings)
# preprocessed_image = img_preprocess(image)


# apply img_preprocess to each train item
X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))

plt.imshow(X_train[random.randint(0, len(X_train)-1)])
plt.axis('off')
# plt.show()
print(X_train.shape)

def nvidia_model():
    model = Sequential()

    model.add(Conv2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3),
                     activation='elu'))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    optimizer = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model



model = nvidia_model()
print(model.summary())

history = model.fit(X_train, y_train, epochs=25,
         validation_data=(X_valid, y_valid), batch_size=100, verbose=1, shuffle=1)

model.save('model.h5')

# model = load_model('model.h5')

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()