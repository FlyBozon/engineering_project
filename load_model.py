import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=4,                      # class num in dataset
)

# from torchgeo.models import resnet18
# import torch

# model = resnet18(weights="sentinel2_all", num_classes=10)