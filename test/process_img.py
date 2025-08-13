#import rasterio
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from load_model import model

class ImageProcessor:
    def __init__(self, image_path, tile_size=512, overlap=64):
        self.image_path = image_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.image = self.load_tiff() 
        self.model = model 

    def load_tiff(self):
        #other options
        # img_cv2=cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # src = rasterio.open(image_path)
        # array = src.read(1)
        return tifffile.imread(self.image_path)

    def split_into_tiles(self):
        h, w = self.image.shape[:2]
        tiles = []
        positions = []
        step = self.tile_size - self.overlap

        for y in range(0, h - self.tile_size + 1, step):
            for x in range(0, w - self.tile_size + 1, step):
                tile = self.image[y:y+self.tile_size, x:x+self.tile_size]
                tiles.append(tile)
                positions.append((x, y))
        return tiles, positions
    
    #for easier testing, to not to process whole tiff image
    def cut_one_tile(self, x_pos, y_pos):
        h, w = self.image.shape[:2]
        if((x_pos+self.tile_size)>w or x_pos<0):
             print("x out of range")
             return 0
        if((y_pos+self.tile_size)>h and y_pos<0):
            print("y out of range")
            return 0
        
        x=x_pos
        y=y_pos

        tile = self.image[y:y+self.tile_size, x:x+self.tile_size]
        pos = (x,y)

        return tile, pos

    def show(self, image=None, is_segmentation=False, oryg_img=None):
        if image is None:
            image = self.image

        if is_segmentation and oryg_img is not None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].imshow(oryg_img, cmap='gray' if oryg_img.ndim == 2 else None)
            axs[0].set_title("Oryginalny obraz")
            axs[0].axis('off')

            im = axs[1].imshow(image, cmap='tab10', vmin=0, vmax=3)
            axs[1].set_title("Segmentacja")
            axs[1].axis('off')
            # plt.imshow(image, cmap='tab10', vmin=0, vmax=3)  # 4 klasy (0-3)
            # plt.colorbar(label='Class ID')

            fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, label='Class ID')
            plt.tight_layout()
            plt.show()

        else:
            plt.imshow(
                image,
                cmap='tab10' if is_segmentation else ('gray' if image.ndim == 2 else None),
                vmin=0 if is_segmentation else None,
                vmax=3 if is_segmentation else None
            )
            if is_segmentation:
                plt.colorbar(label='Class ID')
            plt.axis('off')
            plt.show()


    def preprocess_tile(self, tile):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(tile).unsqueeze(0)

    def predict_tile(self, tile):
        if self.model is None:
            raise ValueError("No model provided.")
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.preprocess_tile(tile)
            if input_tensor.shape[1] != 3:  #ensure 3 channels
                input_tensor = input_tensor.repeat(1, 3, 1, 1)
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        return prediction
    
    def create_weight_mask(self):
        weight = np.ones((self.tile_size, self.tile_size), dtype=np.float32)
        fade = self.overlap // 2

        if fade > 0:
            linear = np.linspace(0.5, 1, fade)
            weight[:fade, :] *= linear[:, None]
            weight[-fade:, :] *= linear[::-1, None]
            weight[:, :fade] *= linear[None, :]
            weight[:, -fade:] *= linear[::-1][None, :]
        return weight

    def merge_predictions(self, predictions, positions, original_shape):
        h, w = original_shape[:2]
        result = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        weight = self.create_weight_mask()

        for pred, (x, y) in zip(predictions, positions):
            weighted_pred = pred.astype(np.float32) * weight
            result[y:y+self.tile_size, x:x+self.tile_size] += weighted_pred
            count_map[y:y+self.tile_size, x:x+self.tile_size] += weight

        normalized_result = result / np.maximum(count_map, 1e-5)
        return normalized_result.astype(np.uint8)
