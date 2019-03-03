import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from sklearn.utils import shuffle

import cv2
import pandas as pd
import ntpath
import random

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
plt.show()


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
plt.show()