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

# code_dir = "/scratch/markryku/engineering_project"
# data_dir = "/data/markryku/"



datasets_info = "datasets_info.json" #f'{code_dir}/datasets_info.json' #"datasets_info.json"
current_dataset ="landcover.ai" #"deepglobe" 

processor = DatasetProcessor(current_dataset, dataset_info_path=datasets_info)
#processor.color_mask_processing()
#processor.plot_img_n_mask("datasets/deepglobe", 20)

processor.into_tiles(256)
processor.choose_useful(0.05)
processor.divide_train_val_test()

processor.setup_model("unet", 'resnet18')

processor.train()  

processor.plot_statistics()