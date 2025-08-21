## tech stack
Tensorflow2.x, OpenCV, NumPy/Matplotlib, Scikit-learn

Segmentation Models - Pre-trained encoder architectures

## pip install
`pip install -r requirements.txt`


### segmentation models
For segmentation-models instalation u can have problems if installing it from pip, as it is not compatible with newer tensorflow version, thats why, u;d better use github version:

`pip uninstall segmentation-models`

`pip install git+https://github.com/qubvel/segmentation_models`


### datasets
info about each dataset is covered in json file - all parameters are read from there at the beginning of the training, so if u want to adjust training parameters for a specific dataset - go there

datasets should be loaded before training, all of them are 1Gb+, so it would take a while 

### Landcover.ai
dataset has 5 classes:
0 - unlabeled, 1 - ..., 2 - ..., ...

after dividing tiles into training patches some of them are fuly unlabeled, so to not to overtrain nn on those unlabeled, they are removed from training dataset (leave only useful ones).


defined order - deepglobe (generic lands), landcoverai (more specific, but less nr of classes), inria (only building/not building classes)

generaly can have problems later, as datasets have around 0.5m/px, when real free available data - 10m/px

### DeepGlobe 
dataset has 3 folders train, val and test. Only train has masks. Every image ends with _sat or _mask depending on what is it. Images are 2448x2448. In train set there are 803 satelite images and same number of masks. After patchifying we would get aroung 74k images => that amoung is enoght to train a nn, so other folders with sat images without masks wouldnt be used. Masks are in rgb, so should be converted to 0-n_classes grayscale values. 

Also for first testing training on kaggle, to not to use too many resources, and simply test if everything works correctly, would use only part of images (around 100/803), which must be enought for testing.