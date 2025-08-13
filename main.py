import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
import os
import random
#later check if all libs needed, as were copyied from old files
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from configuration import *

datasets_info = "datasets_info.json"

current_dataset = "landcover.ai"
dataset_path, n_classes, dataset_size, labels, count, training_params = load_dataset(current_dataset, datasets_info)

#for now would use the same for all, to simplify structure and training
patch_size=256 #or 512 for drone i guess
batch_size = 16 
seed = 42

root_dir = "data/"
OUTPUT_DIR = "output"+dataset_path

# other dirs
os.makedirs(f'{OUTPUT_DIR}/{patch_size}_patches/images', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/{patch_size}_patches/masks', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/useful_patches/images', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/useful_patches/masks', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/data_for_training/train_images/train', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/data_for_training/train_masks/train', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/data_for_training/val_images/val', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/data_for_training/val_masks/val', exist_ok=True)

#random seeds for reproducibility
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)



