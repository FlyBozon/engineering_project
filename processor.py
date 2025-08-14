import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import random
import json
import os

class DatasetProcessor:
    def __init__(self, dataset_name, dataset_info_path="datasets_info.json", input_dir="datasets"):
        self.dataset_name = dataset_name
        self.input_dir = input_dir
        
        # Load dataset info
        self._load_dataset_info(dataset_info_path)
        self._setup_paths()
        self.patch = 256
        self._create_output_dirs()
        
    def _load_dataset_info(self, dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            config = json.load(f)
        
        dataset = config['datasets'][self.dataset_name]
        self.n_classes = dataset['classes']['num_classes']
        self.dataset_dir = dataset['paths']['dataset_dir']
        self.img_format = dataset['data_format']['image_format']
        self.mask_format = dataset['data_format']['mask_format']
        self.training_params = config['training_params']['default']
        #self.patch = config['preprocessing']['patch_size']
        
        print(f"Loaded {dataset['name']}: {self.n_classes} classes")
        
    def _setup_paths(self):
        self.images_path = f"{self.input_dir}/{self.dataset_dir}/images"
        self.masks_path = f"{self.input_dir}/{self.dataset_dir}/masks"
        self.output_dir = f"output_{self.dataset_dir}"
        
        # Get file lists
        self.image_files = glob.glob(f"{self.images_path}/*.{self.img_format}")
        self.mask_files = glob.glob(f"{self.masks_path}/*.{self.mask_format}")
        self.dataset_size = len(self.image_files)
        
    def _create_output_dirs(self):
        # Define all directory paths as instance variables
        self.patches_images_dir = f'{self.output_dir}/{self.patch}_patches/images'
        self.patches_masks_dir = f'{self.output_dir}/{self.patch}_patches/masks'
        self.useful_images_dir = f'{self.output_dir}/useful_patches/images'
        self.useful_masks_dir = f'{self.output_dir}/useful_patches/masks'
        self.train_images_dir = f'{self.output_dir}/data_for_training/train/images'
        self.train_masks_dir = f'{self.output_dir}/data_for_training/train/masks'
        self.val_images_dir = f'{self.output_dir}/data_for_training/val/images'
        self.val_masks_dir = f'{self.output_dir}/data_for_training/val/masks'
        self.test_images_dir = f'{self.output_dir}/data_for_training/test/images'
        self.test_masks_dir = f'{self.output_dir}/data_for_training/test/masks'
        
        
        dirs = [
            self.patches_images_dir,
            self.patches_masks_dir,
            self.useful_images_dir,
            self.useful_masks_dir,
            self.train_images_dir,
            self.train_masks_dir,
            self.val_images_dir,
            self.val_masks_dir,
            self.test_images_dir,
            self.test_masks_dir
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def analyze_sample(self):
        random_img = random.choice(self.image_files)
        random_mask = random.choice(self.mask_files)
        
        temp_img = cv2.imread(random_img)
        # plt.imshow(temp_img[:,:,2])
        # plt.show()
        
        temp_mask = cv2.imread(random_mask, cv2.IMREAD_GRAYSCALE)
        labels, count = np.unique(temp_mask, return_counts=True)
        print("Labels are: ", labels, " and the counts are: ", count)
        
        if len(labels) != self.n_classes:
            print(f"ERROR: Expected {self.n_classes} classes, but found {len(labels)} in mask!")
            
        return labels, count
    
    def into_tiles(self, patch_size, overlap_size=0):
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
                        print(f"Tile already exists, skipping: {patch_name}")
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

    def choose_useful(self, usefulness_percent=0.05): #at least 5% useful area (?)
        # return nr of useful&useless
        # save useful in useful folder
        useless=0
        useful=0

        img_list = os.listdir(self.patches_images_dir)
        msk_list = os.listdir(self.patches_masks_dir)

        for img in range(len(img_list)):   #using t1_list as all lists are of same size
            img_name=img_list[img]
            mask_name = msk_list[img]
            print("Now preparing image and masks number: ", img)
            
            temp_image=cv2.imread(self.patches_images_dir+'/'+img_list[img], 1)
        
            temp_mask=cv2.imread(self.patches_masks_dir+'/'+msk_list[img], 0)
            #temp_mask=temp_mask.astype(np.uint8)
            
            val, counts = np.unique(temp_mask, return_counts=True)
            
            if (1 - (counts[0]/counts.sum())) > usefulness_percent: 
                print("Save Me")
                useful+=1        
                if os.path.exists(self.useful_images_dir+'/'+img_name):
                    print(f"Tile already exists, skipping")  
                    continue
                cv2.imwrite(self.useful_images_dir+'/'+img_name, temp_image)
                cv2.imwrite(self.useful_masks_dir+'/'+mask_name, temp_mask)
                
            else:
                print("I am useless")   
                useless +=1
        
        print(f'Useful = {useful}, useles = {useless}')
        pass

    def divide_train_val_test(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        patch_files = [f for f in os.listdir(self.patches_images_dir) if f.endswith('.png')]
        
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
            os.system(f"cp '{self.patches_images_dir}/{filename}' '{self.train_images_dir}/'")
            os.system(f"cp '{self.patches_masks_dir}/{filename}' '{self.train_masks_dir}/'")
        
        for filename in val_files:
            os.system(f"cp '{self.patches_images_dir}/{filename}' '{self.val_images_dir}/'")
            os.system(f"cp '{self.patches_masks_dir}/{filename}' '{self.val_masks_dir}/'")
        
        for filename in test_files:
            os.system(f"cp '{self.patches_images_dir}/{filename}' '{self.test_images_dir}/'")
            os.system(f"cp '{self.patches_masks_dir}/{filename}' '{self.test_masks_dir}/'")
        
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")  # Fixed syntax
        
    def setup_model(self, model):
        #get info about which model to load (maybe like a string or smth)
        # return model parameters count or some summary?
        pass

    def paramters(self):
        #training parameters for a specific model?
        pass

    def train(self):
        pass

    def save_checkpoint(self, epoch, model, optimizer, metrics):
        #save model progress every x epoch, learning rate scheduler state, random seeds for reproducibility
        pass

    def check_for_not_finished_training(self):
        #return unfinished training parameters if found or false?
        #resume from latest checkpoint, restore optimizer state and epoch number
        pass

    def validate_data_integrity(self, images, masks):
        #check for corrupted files, mismatched image-mask pairs, invalid class IDs
        #plot random img+mask to check if it is ok
        pass

    def convert_mask_into_labels(self):
        #some datasets have rgb masks or other strange types (e.g. deepglobe), 
        # so i want to standartize them in a way like landcoverai has (numbers from 0 to 5)
        # found nice word for that - one-hot encoding
        pass

    def preprocess_image(self):
        # normalize, resize, convert color channels, maybe cloud masking
        pass


    def preprocess_mask(mself):
        # convert color-coded masks to integer class IDs, one-hot encoding if needed.
        pass

    def augment_data(self, images, masks):
        #apply random flips, rotations, brightness changes, noise injection, etc

        #I guess woulnt need that, as the datsets are pretty big
        pass

    def evaluate(self, model, dataloader):
        #compute IoU, precision, recall, F1/Dice.
        pass

    def plot_sample_predictions(self, model, images, masks):
        # visual sanity check of model output during/after training
        pass

    def stitch_tiles(self, predictions, original_image_shape):
        #recombine tiles back into full-size satellite image masks
        pass

    def export_results(self, predictions, output_path):
        #save predictions in georeferenced format (GeoTIFF, etc.) - maybe for later
        pass

    def calculate_class_weights(self, masks):
        # compute inverse frequency weights for imbalanced classes
        pass

    def learning_rate_scheduler(self, optimizer, epoch, metrics):
        #adjust learning rate based on validation performance
        pass

    def early_stopping_check(self, val_metrics, patience):
        #stop training when validation metrics dont change much
        pass

    def post_process_predictions(self, predictions):
        #apply CRF, morphological operations, or filtering to clean up predictions
        pass

    def ensemble_predictions(self, model_list, image):
        #combine predictions from multiple models for better accuracy
        pass