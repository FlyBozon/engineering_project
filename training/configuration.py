import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import random
import json

INPUT_DIR = "datasets"


def load_dataset(dataset_name, dataset_info="datasets_info.json"):
   #load dataset_info - nr of classes, path, img n mask format
   with open(dataset_info, 'r') as f:
       config = json.load(f)
   
   dataset = config['datasets'][dataset_name]
   n_classes = dataset['classes']['num_classes']
   dataset_dir = dataset['paths']['dataset_dir']
   img_format = dataset['data_format']['image_format']
   mask_format = dataset['data_format']['mask_format']
   training_params = config['training_params']['default']
   
   images_path = f"{INPUT_DIR}/{dataset_dir}/images"
   masks_path = f"{INPUT_DIR}/{dataset_dir}/masks"
   
   image_files = glob.glob(f"{images_path}/*.{img_format}")
   mask_files = glob.glob(f"{masks_path}/*.{mask_format}")
   
   random_img = random.choice(image_files)
   random_mask = random.choice(mask_files)
   
   temp_img = cv2.imread(random_img)
   plt.imshow(temp_img[:,:,2])  #view each channel...
   plt.show()
   
   temp_mask = cv2.imread(random_mask, cv2.IMREAD_GRAYSCALE)
   labels, count = np.unique(temp_mask, return_counts=True)
   print("Labels are: ", labels, " and the counts are: ", count)
   
   if len(labels) != n_classes:
       print(f"ERROR: Expected {n_classes} classes, but found {len(labels)} in mask!")
   
   dataset_size = len(image_files)
   
   return dataset_dir, n_classes, dataset_size, labels, count, training_params

def into_tiles(path, size, overlap_size):
    #return tile, also maybe nr of tiles
    img_path = path+"images/"
    pass

def choose_useful(images, usefulness_percent):
    # return nr of useful&useless
    # save useful in useful folder
    pass

def divide_train_val_test(images):
    #create/check if previously created folders
    pass


def setup_model(model):
    #get info about which model to load (maybe like a string or smth)
    # return model parameters count or some summary?
    pass

def paramters():
    #training parameters for a specific model?
    pass

def train():
    pass

def save_checkpoint(epoch, model, optimizer, metrics):
    #save model progress every x epoch, learning rate scheduler state, random seeds for reproducibility
    pass

def check_for_not_finished_training():
    #return unfinished training parameters if found or false?
    #resume from latest checkpoint, restore optimizer state and epoch number
    pass

def validate_data_integrity(images, masks):
    #check for corrupted files, mismatched image-mask pairs, invalid class IDs
    #plot random img+mask to check if it is ok
    pass

def convert_mask_into_labels():
    #some datasets have rgb masks or other strange types (e.g. deepglobe), 
    # so i want to standartize them in a way like landcoverai has (numbers from 0 to 5)
    # found nice word for that - one-hot encoding
    pass

def preprocess_image(image):
    # normalize, resize, convert color channels, maybe cloud masking
    pass


def preprocess_mask(mask):
    # convert color-coded masks to integer class IDs, one-hot encoding if needed.
    pass

def augment_data(images, masks):
    #apply random flips, rotations, brightness changes, noise injection, etc

    #I guess woulnt need that, as the datsets are pretty big
    pass

def evaluate(model, dataloader):
    #compute IoU, precision, recall, F1/Dice.
    pass

def plot_sample_predictions(model, images, masks):
    # visual sanity check of model output during/after training
    pass

def stitch_tiles(predictions, original_image_shape):
    #recombine tiles back into full-size satellite image masks
    pass

def export_results(predictions, output_path):
    #save predictions in georeferenced format (GeoTIFF, etc.) - maybe for later
    pass

def calculate_class_weights(masks):
    # compute inverse frequency weights for imbalanced classes
    pass

def learning_rate_scheduler(optimizer, epoch, metrics):
    #adjust learning rate based on validation performance
    pass

def early_stopping_check(val_metrics, patience):
    #stop training when validation metrics dont change much
    pass

def post_process_predictions(predictions):
    #apply CRF, morphological operations, or filtering to clean up predictions
    pass

def ensemble_predictions(model_list, image):
    #combine predictions from multiple models for better accuracy
    pass