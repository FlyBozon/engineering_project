import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
import os
import random
#later check if all libs needed, as were copyied from old files
import tensorflow as tf
from tensorflow import keras
# import segmentation_models as sm
# from tensorflow.keras.metrics import MeanIoU
# from sklearn.preprocessing import MinMaxScaler
# from keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

from processor import *

datasets_info = "datasets_info.json"
current_dataset = "landcover.ai"

dataset = DatasetProcessor(current_dataset, datasets_info)

patch_size = 256
batch_size = 16 
seed = 42

#random seeds for reproducibility
np.random.seed(seed)
# tf.random.set_seed(seed)
random.seed(seed)

#labels, count = dataset.analyze_sample()

#num_patches = dataset.into_tiles(patch_size, overlap_size=64)
#print(f'Num patches = {num_patches}')

#dataset.plot_img_n_mask(dataset.dataset_dir, 10)

#dataset.choose_useful()
#print(f'{dataset.useful_images_dir}')
dataset.plot_img_n_mask(f'{dataset.output_dir}/useful_patches', 10)

