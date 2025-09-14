import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import random
import json
import os

from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#tf.compat.v1.disable_eager_execution() 
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers.schedules import CosineDecay
import datetime
from clearml import Task, Logger
from tensorflow.keras.callbacks import Callback

import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm

code_dir = "/scratch/markryku/engineering_project"
data_dir = "/data/markryku/"
output_dir = "/data/markryku/output/"

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' 

def save_img(img, name, format, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    is_ok = cv2.imwrite(f"{folder}/{name}.{format}", img)
    if not is_ok:
        print(f"Failed to save image: {folder}/{name}.{format}")


class DatasetProcessor:
    def __init__(self, dataset_name, dataset_info_path="datasets_info.json", input_dir="datasets"):
        self.dataset_name = dataset_name
        self.input_dir = f'{data_dir}{input_dir}' #input_dir
        
        self._load_dataset_info(dataset_info_path)
        self._setup_paths()
        self.patch = 512 #256
        self._create_output_dirs()
        
        self.seed = 24
        self.batch_size = 16
        self.n_epochs = 100
        self.current_version = 1
        self.scaler = MinMaxScaler()
        self.BACKBONE = 'resnet34'  # default backbone
        self.preprocess_input = sm.get_preprocessing(self.BACKBONE)
        
        #model and training objects 
        self.model = None
        self.train_img_gen = None
        self.val_img_gen = None
        self.steps_per_epoch = None
        self.val_steps_per_epoch = None
        self.history = None

        self.tensorboard_dir = f'{self.output_dir}/tensorboard'
        self.class_weight_dict = None

        if len(tf.config.list_physical_devices('GPU')) > 1:
            self.strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {self.strategy.num_replicas_in_sync} GPUs")
        else:
            self.strategy = tf.distribute.get_strategy() 
            print("Using default strategy (single device)")
        
    # def _load_dataset_info(self, dataset_info_path):
    #     with open(dataset_info_path, 'r') as f:
    #         config = json.load(f)
        
    #     dataset = config['datasets'][self.dataset_name]
    #     self.n_classes = dataset['classes']['num_classes']
    #     self.dataset_dir = dataset['paths']['dataset_dir']
    #     self.img_format = dataset['data_format']['image_format']
    #     self.mask_format = dataset['data_format']['mask_format']
    #     self.training_params = config['training_params']['default']
    #     self.ignore_class = dataset['classes']['ignore_class'] 
        
    #     print(f"Loaded {dataset['name']}: {self.n_classes} classes")
        
    def _load_dataset_info(self, dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            config = json.load(f)
        
        self._dataset_config = config
        
        dataset = config['datasets'][self.dataset_name]
        self.n_classes = dataset['classes']['num_classes']
        self.dataset_dir = dataset['paths']['dataset_dir']
        self.img_format = dataset['data_format']['image_format']
        self.mask_format = dataset['data_format']['mask_format']
        self.training_params = config['training_params']['default']
        self.ignore_class = dataset['classes']['ignore_class']
        
        if 'preprocessing' in dataset:
            preprocessing = dataset['preprocessing']
            if 'patch_size' in preprocessing:
                self.patch = preprocessing['patch_size']
                print(f"Using patch size from config: {self.patch}")
            if 'overlap' in preprocessing:
                self.overlap = preprocessing['overlap']
                print(f"Using overlap from config: {self.overlap}")
        
        print(f"Loaded {dataset['name']}: {self.n_classes} classes")
        
        if self.dataset_name == 'uavid':
            self.setup_uavid_color_mapping()

    def _setup_paths(self):
        self.images_path = f"{self.input_dir}/{self.dataset_dir}/images"
        self.masks_path = f"{self.input_dir}/{self.dataset_dir}/masks"
        self.output_dir =f"{output_dir}output_{self.dataset_dir}" #f"output_{self.dataset_dir}" 
        
        self.image_files = glob.glob(f"{self.images_path}/*.{self.img_format}")
        self.mask_files = glob.glob(f"{self.masks_path}/*.{self.mask_format}")
        self.dataset_size = len(self.image_files)
        
    def _create_output_dirs(self):
        self.patches_images_dir = f'{self.output_dir}/{self.patch}_patches/images'
        self.patches_masks_dir = f'{self.output_dir}/{self.patch}_patches/masks'
        # for uavid should rather skip useful patches and just divide all patches into train/val/test, as its drone imagery, so have a lot of details on image
        # self.useful_images_dir = f'{self.output_dir}/useful_patches/images'
        # self.useful_masks_dir = f'{self.output_dir}/useful_patches/masks'
        self.train_images_dir = f'{self.output_dir}/data_for_training/train_images'
        self.train_masks_dir = f'{self.output_dir}/data_for_training/train_masks'
        self.val_images_dir = f'{self.output_dir}/data_for_training/val_images'
        self.val_masks_dir = f'{self.output_dir}/data_for_training/val_masks'
        self.test_images_dir = f'{self.output_dir}/data_for_training/test_images'
        self.test_masks_dir = f'{self.output_dir}/data_for_training/test_masks'
        self.checkpoints_dir = f'{self.output_dir}/checkpoints'
        self.models_dir = f'{self.output_dir}/models'
        
        
        dirs = [
            self.patches_images_dir,
            self.patches_masks_dir,
            # self.useful_images_dir,
            # self.useful_masks_dir,
            self.train_images_dir,
            self.train_masks_dir,
            self.val_images_dir,
            self.val_masks_dir,
            self.test_images_dir,
            self.test_masks_dir,
            f'{self.train_images_dir}/train',
            f'{self.train_masks_dir}/train',
            f'{self.val_images_dir}/val',
            f'{self.val_masks_dir}/val',
            f'{self.test_images_dir}/test',
            f'{self.test_masks_dir}/test',
            self.checkpoints_dir, 
            self.models_dir
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def analyze_sample(self):
        random_img = random.choice(self.image_files)
        random_mask = random.choice(self.mask_files)
        
        temp_img = cv2.imread(random_img)
        if hasattr(self, 'class_colors'):  #uavid and other colorful masks
            temp_mask = cv2.imread(random_mask, cv2.IMREAD_COLOR)
            temp_mask = self.convert_color_mask_to_labels(temp_mask)
        else:  #other
            temp_mask = cv2.imread(random_mask, cv2.IMREAD_GRAYSCALE)

        labels, count = np.unique(temp_mask, return_counts=True)
        print("Labels are: ", labels, " and the counts are: ", count)
        
        if len(labels) != self.n_classes:
            print(f"ERROR: Expected {self.n_classes} classes, but found {len(labels)} in mask!")
            
        return labels, count
    
    def into_tiles(self, patch_size, overlap_size=64, is_drone=False):
        print(f"Creating {patch_size}x{patch_size} patches with {overlap_size} overlap...")
        
        step = patch_size - overlap_size if overlap_size > 0 else patch_size
        
        img_count = self._process_elements(
            self.images_path, 
            f'{self.output_dir}/{patch_size}_patches/images',
            self.img_format, 
            patch_size, 
            step, 
            isImage=True
        )

        mask_count = self._process_elements(
            self.masks_path,
            f'{self.output_dir}/{patch_size}_patches/masks', 
            self.mask_format,
            patch_size,
            step,
            isImage=False
        )
        
        print(f"Created {img_count} image patches and {mask_count} mask patches")
        return img_count
        
    def patchify(self, image, patch_shape, step):
        if len(image.shape) == 2:  #grayscale image/mask
            img_h, img_w = image.shape
            patch_h, patch_w = patch_shape
            
            n_patches_h = (img_h - patch_h) // step + 1
            n_patches_w = (img_w - patch_w) // step + 1
            
            patches = np.zeros((n_patches_h, n_patches_w, patch_h, patch_w), dtype=image.dtype)
            
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    start_h = i * step
                    start_w = j * step
                    end_h = start_h + patch_h
                    end_w = start_w + patch_w
                    
                    patches[i, j] = image[start_h:end_h, start_w:end_w]
                    
        elif len(image.shape) == 3:  #color img
            img_h, img_w, img_c = image.shape
            patch_h, patch_w, patch_c = patch_shape
            
            n_patches_h = (img_h - patch_h) // step + 1
            n_patches_w = (img_w - patch_w) // step + 1
            
            patches = np.zeros((n_patches_h, n_patches_w, 1, patch_h, patch_w, patch_c), dtype=image.dtype)
            
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    start_h = i * step
                    start_w = j * step
                    end_h = start_h + patch_h
                    end_w = start_w + patch_w
                    
                    patches[i, j, 0] = image[start_h:end_h, start_w:end_w, :]
        return patches

    def _process_elements(self, input_path, output_path, file_format, patch_size, step, isImage=True):
        elems = os.listdir(input_path)
        tile_count = 0

        matching_files = [elem for elem in elems if elem.endswith(file_format)]
        
        if not matching_files:
            print(f"No .{file_format} files found in {input_path}")
            return 0

        for elem_name in matching_files:
            print(f"Processing {elem_name}")
            
            if isImage:
                elem = cv2.imread(f"{input_path}/{elem_name}", 1)
                if elem is None:
                    print(f"Could not read image: {elem_name}")
                    continue
                h, w = elem.shape[:2]
                
                if step < patch_size:  # overlap
                    patches = self.patchify(elem, (patch_size, patch_size, 3), step) 
                else:  # no overlap - crop
                    SIZE_X = (w // patch_size) * patch_size
                    SIZE_Y = (h // patch_size) * patch_size
                    elem = elem[:SIZE_Y, :SIZE_X]
                    patches = self.patchify(elem, (patch_size, patch_size, 3), patch_size)
            else:
                if hasattr(self, 'class_colors'):  #uavid and other colorful masks
                    elem = cv2.imread(f"{input_path}/{elem_name}", cv2.IMREAD_COLOR)
                    elem = self.convert_color_mask_to_labels(elem)  
                else:  #other
                    elem = cv2.imread(f"{input_path}/{elem_name}", 0)
                if elem is None:
                    print(f"Could not read mask: {elem_name}")
                    continue
                h, w = elem.shape[:2]
                
                if step < patch_size:  # overlap
                    patches = self.patchify(elem, (patch_size, patch_size), step)  
                else:  # no overlap - crop
                    SIZE_X = (w // patch_size) * patch_size
                    SIZE_Y = (h // patch_size) * patch_size
                    elem = elem[:SIZE_Y, :SIZE_X]
                    patches = self.patchify(elem, (patch_size, patch_size), patch_size)
            
            base_name = elem_name.split('.')[0]
            for row in range(patches.shape[0]):
                for col in range(patches.shape[1]):
                    if isImage:
                        single_patch = patches[row, col, 0]
                    else:
                        single_patch = patches[row, col]
                    
                    patch_name = f"{base_name}_patch_{row}_{col}.png"
                    output_file_path = f"{output_path}/{patch_name}"
                    if os.path.exists(output_file_path):
                        #print(f"Tile already exists, skipping: {patch_name}")
                        tile_count += 1  
                        continue
                    success = cv2.imwrite(output_file_path, single_patch)
                    if not success:
                        print(f"Failed to save: {output_file_path}")
                    else:
                        tile_count += 1
        
        return tile_count

    def plot_img_n_mask(self, folder_dir, n=3):
        images_dir = f"{folder_dir}/images"
        masks_dir = f"{folder_dir}/masks"
        
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            return
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.tif', '.jpg'))]
        
        if len(image_files) == 0:
            print("No images found")
            return
        
        selected_images = random.sample(image_files, min(n, len(image_files)))
        
        for i in range(n):
            img_name = selected_images[i]
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            img_path = f"{images_dir}/{img_name}"
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[0].imshow(img_rgb)
                axes[0].set_title(f"Image: {img_name}")
                axes[0].axis('off')
            
            mask_path = f"{masks_dir}/{img_name}"
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    axes[1].imshow(mask, cmap='tab10')
                    axes[1].set_title(f"Mask: {img_name}")
                    axes[1].axis('off')
                    
                    labels = np.unique(mask)
                    print(f"{img_name}: labels {labels}")
            else:
                axes[1].text(0.5, 0.5, 'No mask', ha='center', va='center')
                axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()

    def choose_useful(self, usefulness_percent=0.05): #at least 5% useful area 
        useless = 0
        useful = 0

        img_list = os.listdir(self.patches_images_dir)
        msk_list = os.listdir(self.patches_masks_dir)

        for img in range(len(img_list)):   
            img_name = img_list[img]
            mask_name = msk_list[img]
            print("Now preparing image and masks number: ", img)
            
            temp_image = cv2.imread(self.patches_images_dir+'/'+img_list[img], 1)
            temp_mask = cv2.imread(self.patches_masks_dir+'/'+msk_list[img], 0)
            
            val, counts = np.unique(temp_mask, return_counts=True)
            if self.ignore_class is not None:
                ignore = self.ignore_class
            else: 
                ignore = 0
            if (1 - (counts[ignore]/counts.sum())) > usefulness_percent: 
                print("Save Me")
                useful += 1        
                if os.path.exists(self.useful_images_dir+'/'+img_name):
                    print(f"Tile already exists, skipping")  
                    continue
                cv2.imwrite(self.useful_images_dir+'/'+img_name, temp_image)
                cv2.imwrite(self.useful_masks_dir+'/'+mask_name, temp_mask)
                
            else:
                print("I am useless")   
                useless += 1
        
        print(f'Useful = {useful}, useless = {useless}')

    def divide_train_val_test(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        image_files = set(f for f in os.listdir(self.useful_images_dir) if f.endswith('.png'))
        mask_files = set(f for f in os.listdir(self.useful_masks_dir) if f.endswith('.png'))
        
        patch_files = list(image_files.intersection(mask_files))
        
        if len(patch_files) == 0:
            print("No patches found!")
            return
        
        random.shuffle(patch_files)
        
        total = len(patch_files)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        train_files = patch_files[:train_end]
        val_files = patch_files[train_end:val_end]
        test_files = patch_files[val_end:]
        
        for filename in train_files:
            os.system(f"cp '{self.useful_images_dir}/{filename}' '{self.train_images_dir}/train/'")
            os.system(f"cp '{self.useful_masks_dir}/{filename}' '{self.train_masks_dir}/train/'")
        
        for filename in val_files:
            os.system(f"cp '{self.useful_images_dir}/{filename}' '{self.val_images_dir}/val/'")
            os.system(f"cp '{self.useful_masks_dir}/{filename}' '{self.val_masks_dir}/val/'")
        
        for filename in test_files:
            os.system(f"cp '{self.useful_images_dir}/{filename}' '{self.test_images_dir}/test/'")
            os.system(f"cp '{self.useful_masks_dir}/{filename}' '{self.test_masks_dir}/test/'")
        
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

        if len(image_files) != len(mask_files):
            print(f"Warning: {len(image_files)} images but {len(mask_files)} masks in useful directories")
            
    def calculate_class_weights(self, masks_dir=None):
        if masks_dir is None:
            masks_dir = self.train_masks_dir+"/train"
        
        print(f"Class weights from {masks_dir}...")
        
        if not os.path.exists(masks_dir):
            print(f"dir {masks_dir} doesnt exist.")
            return None

        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
        if not mask_files:
            print("No mask files found :(")
            return None

        class_pixel_counts = {}
        total_pixels = 0

        for mask_file in mask_files:
            mask_path = os.path.join(masks_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Warning: couldnt read {mask_file}. Skipping.")
                continue

            if mask.size == 0:
                print(f"Warning: empty mask {mask_file}. Skipping.")
                continue

            labels, counts = np.unique(mask, return_counts=True)
            for label, count in zip(labels, counts):
                class_pixel_counts[label] = class_pixel_counts.get(label, 0) + count
                total_pixels += count

        if total_pixels == 0:
            print("ERROR: Total pixels = 0. Check your mask files.")
            return None

        class_weights = {
            label: (1.0 / (count / total_pixels))
            for label, count in class_pixel_counts.items()
        }

        #normalize
        avg_weight = np.mean(list(class_weights.values()))
        class_weights = {label: w / avg_weight for label, w in class_weights.items()}

        print("class weights:")
        for label, weight in class_weights.items():
            count = class_pixel_counts[label]
            percentage = (count / total_pixels) * 100
            print(f"Class {label}: weight={weight:.4f}, pixels={count:,} ({percentage:.2f}%)")

        return class_weights
    
    # additional preprocessing after datagen
    def preprocess_data(self, img, mask):
        # img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        # img = self.preprocess_input(img)  # Preprocess based on the pretrained backbone
        
        #normalize images to 0-1 range manually (instead of MinMaxScaler)
        img = img.astype(np.float32) / 255.0
        
        img = self.preprocess_input(img)
        
        mask = to_categorical(mask, self.n_classes).astype(np.float32)
        
        return (img, mask)

    # def setup_model(self, architecture="unet", backbone='resnet34'):
    #     print(f"Setting up {architecture.upper()} model with {backbone} backbone...")
        
    #     self.architecture = architecture
    #     self.BACKBONE = backbone
    #     self.preprocess_input = sm.get_preprocessing(self.BACKBONE)
        
    #     num_train_imgs = len(os.listdir(f"{self.train_images_dir}/train"))
    #     num_val_images = len(os.listdir(f"{self.val_images_dir}/val"))

    #     print(f'num_training_imgs = {num_train_imgs}, num_val_images = {num_val_images}')
        
    #     if num_train_imgs == 0 or num_val_images == 0:
    #         print("ERROR: No training or validation images found!")
    #         return
        
    #     self.steps_per_epoch = num_train_imgs // self.batch_size
    #     self.val_steps_per_epoch = num_val_images // self.batch_size
        
    #     IMG_HEIGHT = self.patch
    #     IMG_WIDTH = self.patch
    #     IMG_CHANNELS = 3

    #     # Select architecture
    #     model_args = {
    #         'backbone_name': self.BACKBONE,
    #         'encoder_weights': 'imagenet',
    #         'input_shape': (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
    #         'classes': self.n_classes,
    #         'activation': 'softmax'
    #     }
        
    #     if architecture.lower() == 'unet':
    #         self.model = sm.Unet(**model_args)
    #     elif architecture.lower() == 'linknet':
    #         self.model = sm.Linknet(**model_args)
    #     elif architecture.lower() == 'fpn':
    #         self.model = sm.FPN(**model_args)
    #     elif architecture.lower() == 'pspnet':
    #         self.model = sm.PSPNet(**model_args)
    #     elif architecture.lower() == 'deeplabv3':
    #         self.model = sm.DeepLabV3(**model_args)
    #     elif architecture.lower() == 'deeplabv3plus':
    #         self.model = sm.DeepLabV3Plus(**model_args)
    #     else:
    #         print(f"ERROR: Unknown architecture '{architecture}'. Using U-Net as default.")
    #         self.model = sm.Unet(**model_args)
        
    #     # Compile model
    #     self.model.compile(
    #         optimizer='adam',
    #         loss='categorical_crossentropy', 
    #         metrics=['accuracy', sm.metrics.iou_score]
    #     )

    #     print("Model compiled successfully!")
    #     print(f"Architecture: {architecture.upper()}")
    #     print(f"Model input shape: {self.model.input_shape}")
    #     print(f"Model parameters: {self.model.count_params():,}")
    #     print(f"Steps per epoch: {self.steps_per_epoch}")
    #     print(f"Validation steps: {self.val_steps_per_epoch}")

    def setup_model(self, architecture="fpn", backbone='efficientnet-b3'):
        with self.strategy.scope():
            print(f"Setting up {architecture.upper()} model with {backbone} backbone...")
            
            self.architecture = architecture
            self.BACKBONE = backbone
            self.preprocess_input = sm.get_preprocessing(self.BACKBONE)
            
            # Calculate steps (adjust for multi-GPU)
            num_train_imgs = len(os.listdir(f"{self.train_images_dir}/train"))
            num_val_images = len(os.listdir(f"{self.val_images_dir}/val"))
            
            # Adjust batch size for multiple GPUs
            global_batch_size = self.batch_size * self.strategy.num_replicas_in_sync
            self.steps_per_epoch = num_train_imgs // global_batch_size
            self.val_steps_per_epoch = num_val_images // global_batch_size
            
            # Model creation code remains the same...
            IMG_HEIGHT = self.patch
            IMG_WIDTH = self.patch
            IMG_CHANNELS = 3

            model_args = {
                'backbone_name': self.BACKBONE,
                'encoder_weights': 'imagenet',
                'input_shape': (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                'classes': self.n_classes,
                'activation': 'softmax'
            }
            
            if architecture.lower() == 'fpn':
                self.model = sm.FPN(**model_args)
            # ... other architectures
            
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy', 
                metrics=['accuracy', sm.metrics.iou_score]
            )
            if hasattr(self.strategy, 'num_replicas_in_sync'):
                global_batch_size = self.batch_size * self.strategy.num_replicas_in_sync
                print(f"Global batch size: {global_batch_size}")

        print(f"Model will train on {self.strategy.num_replicas_in_sync} GPUs")
        print(f"Global batch size: {global_batch_size}")

    def create_train_generator(self):
        print("Creating training data generators...")
        
        img_data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect'
        )
        
        image_datagen = ImageDataGenerator(**img_data_gen_args)
        mask_datagen = ImageDataGenerator(**img_data_gen_args)
        
        image_generator = image_datagen.flow_from_directory(
            self.train_images_dir,
            class_mode=None,
            batch_size=self.batch_size,
            seed=self.seed,
            target_size=(self.patch, self.patch)
        )
        
        mask_generator = mask_datagen.flow_from_directory(
            self.train_masks_dir,
            class_mode=None,
            color_mode='grayscale',
            batch_size=self.batch_size,
            seed=self.seed,
            target_size=(self.patch, self.patch)
        )
        
        val_image_generator = image_datagen.flow_from_directory(
            self.val_images_dir,
            class_mode=None,
            batch_size=self.batch_size,
            seed=self.seed,
            target_size=(self.patch, self.patch)
        )
        
        val_mask_generator = mask_datagen.flow_from_directory(
            self.val_masks_dir,
            class_mode=None,
            color_mode='grayscale',
            batch_size=self.batch_size,
            seed=self.seed,
            target_size=(self.patch, self.patch)
        )
        
        def train_gen():
            for (img, mask) in zip(image_generator, mask_generator):
                img, mask = self.preprocess_data(img, mask)
                yield (img, mask)
                
        def val_gen():
            for (img, mask) in zip(val_image_generator, val_mask_generator):
                img, mask = self.preprocess_data(img, mask)
                yield (img, mask)
        
        self.train_img_gen = train_gen()
        self.val_img_gen = val_gen()
        
        print("Data generators created successfully!")

    def setup_lr_schedule(self, schedule_type='plateau'):
        if schedule_type == 'plateau':
            return ReduceLROnPlateau(
                monitor='val_iou_score',
                factor=0.2,
                patience=7,
                min_lr=1e-8,
                mode='max',
                verbose=1
            )
        elif schedule_type == 'exponential':
            return ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=self.steps_per_epoch * 8,
                decay_rate=0.92,
                staircase=True
            )
        elif schedule_type == 'cosine':
            return CosineDecay(
                initial_learning_rate=0.001,
                decay_steps=self.steps_per_epoch * self.n_epochs
            )
        
    def set_training_callbacks(self, early_patience=15, lr_patience=7, lr_factor=0.2):
        self.early_patience = early_patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        print(f"Updated training callbacks: early_patience={early_patience}, lr_patience={lr_patience}")

    def train(self):
        if self.model is None:
            print("ERROR: Model not initialized. Call setup_model() first!")
            return
            
        if self.train_img_gen is None:
            print("Creating data generators...")
            self.create_train_generator()
        
        print("Calc class weights...")
        self.apply_class_weights()

        print("Setting up ClearML...")
        self.setup_clearml()

        # print("Re-compiling model with weighted loss...")
        # self.model.compile(
        #     optimizer='adam',
        #     loss=weighted_categorical_crossentropy(self.class_weight_dict),
        #     metrics=['accuracy', custom_iou_metric]
        # )
        # print("Model re-compiled with class weights!")

        print("Re-compiling model with weighted loss...")
        self.model.compile(
            optimizer='adam',
            loss=weighted_categorical_crossentropy(self.class_weight_dict),
            metrics=['accuracy', sm.metrics.iou_score] 
        )
        print("Model re-compiled with class weights!")
        
        callbacks = []
        
        clearml_cb = self.setup_clearml_callback()
        callbacks.append(clearml_cb)

        tensorboard_cb = self.setup_tensorboard()
        callbacks.append(tensorboard_cb)
        
        checkpoint_path = f"{self.checkpoints_dir}/best_model.keras"
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        checkpoint_cb = ModelCheckpoint(
            checkpoint_path,
            monitor='val_iou_score',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint_cb)
        
        early_stop_cb = EarlyStopping(
            monitor='val_iou_score',
            patience=15, #10,
            mode='max',
            verbose=1,
            restore_best_weights=True,
            min_delta=0.001  #improvements > 0.1%
        )
        callbacks.append(early_stop_cb)

        # lr_reduce_cb = ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.5,
        #     patience=5,
        #     min_lr=1e-7,
        #     verbose=1
        # )

        lr_reduce_cb = ReduceLROnPlateau(
            monitor='val_iou_score',  
            factor=0.2,  
            patience=7, 
            min_lr=1e-8,
            mode='max', 
            verbose=1
        )
        callbacks.append(lr_reduce_cb)
        
        print(f"Starting training for {self.n_epochs} epochs...")
        
        self.history = self.model.fit(
            self.train_img_gen,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.n_epochs,
            verbose=1,
            validation_data=self.val_img_gen,
            validation_steps=self.val_steps_per_epoch,
            callbacks=callbacks
            #class_weight=self.class_weight_dict  
        )
        
        #model_filename = f'{self.dataset_name}_{self.n_epochs}_epochs_{self.BACKBONE}_backbone_batch{self.batch_size}_v{self.current_version}.keras'
        actual_epochs = len(self.history.history['loss']) if self.history else self.n_epochs
        model_filename = f'{self.dataset_name}_{actual_epochs}_epochs_{self.BACKBONE}_backbone_batch{self.batch_size}_v{self.current_version}.keras'
        self.model.save(model_filename)
        print(f"Final model saved as: {model_filename}")
        print(f"Best model saved as: {checkpoint_path}")

        if hasattr(self, 'task'):
            self.task.upload_artifact(
                name='final_model',
                artifact_object=model_filename
            )
            
            if self.history:
                self.plot_statistics()  
            print("Model and artifacts uploaded to ClearML")

    def plot_statistics(self):
        if self.history is None:
            print("No training history found. Train the model first!")
            return
            
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        acc = self.history.history['iou_score']
        val_acc = self.history.history['val_iou_score']

        plt.subplot(1, 2, 2)
        plt.plot(epochs, acc, 'y', label='Training IoU')
        plt.plot(epochs, val_acc, 'r', label='Validation IoU')
        plt.title('Training and validation IoU')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def save_checkpoint(self, epoch, model, optimizer, metrics):
        checkpoint_dir = self.checkpoints_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.keras"
        model.save(checkpoint_path)
        
        checkpoint_info = {
            'epoch': epoch,
            'dataset_name': self.dataset_name,
            'backbone': self.BACKBONE,
            'batch_size': self.batch_size,
            'metrics': metrics
        }
        
        info_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}_info.json"
        with open(info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
            
        print(f"Checkpoint saved at epoch {epoch}")

    def check_for_unfinished_training(self):
        checkpoint_dir = self.checkpoints_dir
        
        if not os.path.exists(checkpoint_dir):
            print("No checkpoints found")
            return False
            
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.keras')]
        
        if not checkpoint_files:
            print("No checkpoint files found")
            return False
            
        #find latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        latest_epoch = int(latest_checkpoint.split('_')[2].split('.')[0])
        
        checkpoint_path = f"{checkpoint_dir}/{latest_checkpoint}"
        info_path = f"{checkpoint_dir}/checkpoint_epoch_{latest_epoch}_info.json"
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                checkpoint_info = json.load(f)
                
            print(f"Found checkpoint at epoch {latest_epoch}")
            print(f"Checkpoint info: {checkpoint_info}")
            
            return {
                'checkpoint_path': checkpoint_path,
                'epoch': latest_epoch,
                'info': checkpoint_info
            }
        else:
            print(f"Found checkpoint file but no info file: {latest_checkpoint}")
            return {
                'checkpoint_path': checkpoint_path,
                'epoch': latest_epoch,
                'info': None
            }

    def set_training_parameters(self, epochs=None, batch_size=None, backbone=None):
        if epochs is not None:
            self.n_epochs = epochs
            print(f"Set epochs to: {self.n_epochs}")
            
        if batch_size is not None:
            self.batch_size = batch_size
            print(f"Set batch size to: {self.batch_size}")
            
        if backbone is not None:
            self.BACKBONE = backbone
            self.preprocess_input = sm.get_preprocessing(self.BACKBONE)
            print(f"Set backbone to: {self.BACKBONE}")


    
    #for landcoverai
    # def apply_class_weights(self):
    #     class_weights = self.calculate_class_weights()
    #     if class_weights is not None:
    #         self.class_weight_dict = {i: class_weights.get(i, 1.0) for i in range(self.n_classes)}
    #         print(f"applied class weights: {self.class_weight_dict}")
    #         return self.class_weight_dict
    #     return None
    def apply_class_weights(self, ignore_classes=None):
        if ignore_classes is None and hasattr(self, '_dataset_config'):
            dataset_config = self._dataset_config['datasets'][self.dataset_name]
            ignore_classes = dataset_config['classes'].get('ignored_classes', [])

        class_weights = self.calculate_class_weights()
        if class_weights is not None:
            #if none is given, then ignore 0 by default
            for ignore_class in ignore_classes:
                if ignore_class in class_weights:
                    class_weights[ignore_class] = 0.0
            
            self.class_weight_dict = {i: class_weights.get(i, 1.0) for i in range(self.n_classes)}
            print(f"Applied class weights with ignored classes: {self.class_weight_dict}")
            print(self.class_weight_dict)
            return self.class_weight_dict
        return None
    
    def setup_tensorboard(self):
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"{self.tensorboard_dir}/{self.dataset_name}_{self.BACKBONE}_{timestamp}"
        
        self.tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,  
            write_graph=True, 
            write_images=True,  
            update_freq='epoch'  
        )
        
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print(f"To view: tensorboard --logdir {log_dir}")
        return self.tensorboard_callback
    
    def setup_clearml(self, project_name="Segmentation", task_name=None):
        if task_name is None:
            task_name = f"{self.dataset_name}_{self.BACKBONE}_patch{self.patch}"
        
        self.task = Task.init(
            project_name=project_name,
            task_name=task_name,
            auto_connect_frameworks=True,  
            auto_connect_arg_parser=True   
        )
        
        self.task.connect({
            'dataset_name': self.dataset_name,
            'backbone': self.BACKBONE,
            'patch_size': self.patch,
            'batch_size': self.batch_size,
            'epochs': self.n_epochs,
            'n_classes': self.n_classes,
            'seed': self.seed,
            'class_weights': self.class_weight_dict if self.class_weight_dict else None
        })
        
        self.logger = self.task.get_logger()
        print(f"ClearML task initialized: {task_name}")
        return self.task
    
    def setup_clearml_callback(self):
        try:
            from clearml.binding.keras_bind import KerasCallback
            return KerasCallback()
        except ImportError:
            print("Warning: ClearML Keras binding not available, using basic callback")

            class ClearMLCallback(Callback):
                def __init__(self, logger):
                    super().__init__()
                    self.logger = logger
                    
                def on_epoch_end(self, epoch, logs=None):
                    if logs and self.logger:
                        for metric_name, value in logs.items():
                            self.logger.report_scalar(
                                title="Training",
                                series=metric_name,
                                value=value,
                                iteration=epoch
                            )
            
            return ClearMLCallback(self.logger if hasattr(self, 'logger') else None)
        
    def log_to_clearml(self, epoch, logs):
        if hasattr(self, 'logger'):
            # Log custom metrics
            for metric_name, value in logs.items():
                self.logger.report_scalar(
                    title="Training Metrics",
                    series=metric_name,
                    value=value,
                    iteration=epoch
                )
            
            # Log class distribution if available
            if self.class_weight_dict:
                for class_id, weight in self.class_weight_dict.items():
                    self.logger.report_scalar(
                        title="Class Weights",
                        series=f"class_{class_id}",
                        value=weight,
                        iteration=0
                    )

    def setup_uavid_color_mapping(self):
        """Setup UAVid color mapping from JSON configuration"""
        if not hasattr(self, 'dataset_name') or self.dataset_name != 'uavid':
            print("Warning: This function is designed for UAVid dataset")
            return
        
        # Get color mapping from loaded dataset info
        if not hasattr(self, '_dataset_config'):
            print("Error: Dataset configuration not loaded. Call _load_dataset_info() first")
            return
        
        dataset_config = self._dataset_config['datasets'][self.dataset_name]
        class_colors_dict = dataset_config['classes']['class_colors']
        class_names = dataset_config['classes']['class_names']
        class_ids = dataset_config['classes']['class_ids']
        
        # Create mapping from class_id to color
        self.class_colors = {}
        for i, (class_name, class_id) in enumerate(zip(class_names, class_ids)):
            self.class_colors[class_id] = class_colors_dict[class_name]
        
        # Create reverse mapping for faster lookup
        self.color_to_class = {}
        for class_id, color in self.class_colors.items():
            color_key = tuple(color)
            self.color_to_class[color_key] = class_id
        
        print(f"UAVid color mapping loaded: {len(self.class_colors)} classes")
        print("Class mapping:")
        for class_id, color in self.class_colors.items():
            class_name = class_names[class_id]
            print(f"  {class_id}: {class_name} -> {color}")


    def convert_color_mask_to_labels(self, color_mask):
        if not hasattr(self, 'class_colors'):
            self.setup_uavid_color_mapping()
        
        label_mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)
        
        if len(color_mask.shape) == 3:
            color_mask_rgb = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
        else:
            color_mask_rgb = color_mask
        
        for class_id, color in self.class_colors.items():
            matches = np.all(color_mask_rgb == color, axis=-1)
            label_mask[matches] = class_id
        
        total_pixels = label_mask.size
        matched_pixels = np.sum(label_mask >= 0)  #all pixels should be matched
        
        if matched_pixels < total_pixels:
            #unique colors that weren't matched
            unique_colors = np.unique(color_mask_rgb.reshape(-1, 3), axis=0)
            known_colors = set(tuple(color) for color in self.class_colors.values())
            unknown_colors = [tuple(color) for color in unique_colors if tuple(color) not in known_colors]
            
            if unknown_colors:
                print(f"Warning: Found {len(unknown_colors)} unknown colors, assigning to ignore class ({self.ignore_class})")
                #unknown colors to ignore class
                for unknown_color in unknown_colors:
                    matches = np.all(color_mask_rgb == unknown_color, axis=-1)
                    label_mask[matches] = self.ignore_class
        
        return label_mask


        #already divided into train/val/test
        #convert color masks to label masks
        #move to correct folders (expected by datagen)
    def uavid_data_preprocess(self):
        print("UAVid data preprocessing...")
        #uavid dataset structure: uavid_train, uavid_val, uavid_test
        #train and val contains seqXX/Images/ and seqXX/Labels/, test has only images
        split_mapping = {
            'uavid_train': 'train', 
            'uavid_val': 'val'
        }

        for uavid_split, standard_split in split_mapping.items():
            print(f"\nProcessing {uavid_split} -> {standard_split} split...")
            
            split_images_path = f"{self.input_dir}/{self.dataset_dir}/{uavid_split}"
            
            split_output_images = f"{self.output_dir}/data_for_training/{standard_split}_images/{standard_split}"
            split_output_masks = f"{self.output_dir}/data_for_training/{standard_split}_masks/{standard_split}"
            
            os.makedirs(split_output_images, exist_ok=True)
            os.makedirs(split_output_masks, exist_ok=True)
            
            if not os.path.exists(split_images_path):
                print(f"  Warning: {split_images_path} does not exist, skipping...")
                continue
            
            #split/seqXX/Images/ and split/seqXX/Labels/
            try:
                seq_dirs = [d for d in os.listdir(split_images_path) if d.startswith('seq')]
            except OSError as e:
                print(f"  Error reading {split_images_path}: {e}")
                continue
                
            if not seq_dirs:
                print(f"  No sequence directories found in {split_images_path}")
                continue
            
            step = self.patch - self.overlap if hasattr(self, 'overlap') and self.overlap > 0 else self.patch
            
            for seq_dir in seq_dirs:
                print(f"  Processing {seq_dir}...")
                
                images_dir = f"{split_images_path}/{seq_dir}/Images"
                labels_dir = f"{split_images_path}/{seq_dir}/Labels"
                
                if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                    print(f"    Skipping {seq_dir}: missing Images or Labels directory")
                    continue
                
                image_files = glob.glob(f"{images_dir}/*.{self.img_format}")
                
                for img_file in image_files:
                    try:
                        img_name = os.path.basename(img_file)
                        base_name = os.path.splitext(img_name)[0]
                        mask_file = f"{labels_dir}/{base_name}.{self.mask_format}"
                        
                        if not os.path.exists(mask_file):
                            print(f"    Warning: No matching mask for {img_name}")
                            continue
                        
                        img = cv2.imread(img_file)
                        color_mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)
                        
                        if img is None or color_mask is None:
                            print(f"    Could not read {img_name} or its mask")
                            continue
                        
                        label_mask = self.convert_color_mask_to_labels(color_mask)
                        
                        h, w = img.shape[:2]
                        
                        n_patches_h = (h - self.patch) // step + 1
                        n_patches_w = (w - self.patch) // step + 1
                        
                        for i in range(n_patches_h):
                            for j in range(n_patches_w):
                                start_h = i * step
                                start_w = j * step
                                end_h = start_h + self.patch
                                end_w = start_w + self.patch
                                
                                img_patch = img[start_h:end_h, start_w:end_w]
                                mask_patch = label_mask[start_h:end_h, start_w:end_w]
                                
                                patch_name = f"{seq_dir}_{base_name}_patch_{i}_{j}.png"
                                
                                img_success = cv2.imwrite(f"{split_output_images}/{patch_name}", img_patch)
                                mask_success = cv2.imwrite(f"{split_output_masks}/{patch_name}", mask_patch)
                                
                                if not img_success or not mask_success:
                                    print(f"    Failed to save patch: {patch_name}")
                        
                        print(f"    Created {n_patches_h * n_patches_w} patches from {img_name}")
                        
                    except Exception as e:
                        print(f"    Error processing {img_file}: {e}")
            
            print(f"  Completed {uavid_split} -> {standard_split} split")
        
        train_imgs = len(os.listdir(f"{self.output_dir}/data_for_training/train_images/train")) if os.path.exists(f"{self.output_dir}/data_for_training/train_images/train") else 0
        train_masks = len(os.listdir(f"{self.output_dir}/data_for_training/train_masks/train")) if os.path.exists(f"{self.output_dir}/data_for_training/train_masks/train") else 0
        val_imgs = len(os.listdir(f"{self.output_dir}/data_for_training/val_images/val")) if os.path.exists(f"{self.output_dir}/data_for_training/val_images/val") else 0
        val_masks = len(os.listdir(f"{self.output_dir}/data_for_training/val_masks/val")) if os.path.exists(f"{self.output_dir}/data_for_training/val_masks/val") else 0
        
        print(f"\nProcessing summary:")
        print(f"  Train: {train_imgs} images, {train_masks} masks")
        print(f"  Val: {val_imgs} images, {val_masks} masks")
        
        if train_imgs == 0 or train_masks == 0:
            print("  Warning: No training data found!")
        if val_imgs == 0 or val_masks == 0:
            print("  Warning: No validation data found!")
        
        print("UAVid data preprocessing completed!")
        print("Data is now ready for training with create_train_generator()")

def custom_iou_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)

    intersection = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    union = tf.reduce_sum(tf.cast(y_true + y_pred, tf.float32)) - intersection

    return intersection / (union + tf.keras.backend.epsilon())

def weighted_categorical_crossentropy(class_weights):
    def loss_function(y_true, y_pred):
        weights_tensor = tf.constant([class_weights[i] for i in range(len(class_weights))], dtype=tf.float32)

        y_true_indices = tf.argmax(y_true, axis=-1)

        pixel_weights = tf.gather(weights_tensor, y_true_indices)

        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        weighted_cce = cce * pixel_weights

        return tf.reduce_mean(weighted_cce)
    

def check_gpu_usage():
    print("Available GPUs:")
    for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
        print(f"  GPU {i}: {gpu}")
    
    print("GPU memory info:")
    for i in range(len(tf.config.list_physical_devices('GPU'))):
        try:
            memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
            print(f"  GPU {i}: {memory_info['current'] / 1024**3:.2f} GB used")
        except:
            print(f"  GPU {i}: Memory info not available")
