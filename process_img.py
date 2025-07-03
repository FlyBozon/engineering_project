import rasterio
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_path, tile_size=512, overlap=64):
        self.image_path = image_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.image = self.load_tiff()  # Load on init

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

    def show(self, image=None):
        if image is None:
            image = self.image
        plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
        plt.axis('off')
        plt.show()

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
