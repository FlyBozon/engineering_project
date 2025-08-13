# Hello world

### Landcover.ai
dataset has 5 classes:
0 - unlabeled, 1 - ..., 2 - ..., ...

after dividing tiles into training patches some of them are fuly unlabeled, so to not to overtrain nn on those unlabeled, they are removed from training dataset (leave only useful ones).

### other datasets


## general structure of the code
0. if started from the beginning check for not finished versions
1. create all dir
2. load dataset
3. prepare for training (into tiles, divide to train/val/test, choose useful patches)
4. load model -> adjust parameters -> ...
5. train (save from time to time)



clearML credentials?

defined order - deepglobe (generic lands), landcoverai (more specific, but less nr of classes), inria (only building/not building classes)

generaly can have problems later, as datasets have around 0.5m/px, when real free available data - 10m/px