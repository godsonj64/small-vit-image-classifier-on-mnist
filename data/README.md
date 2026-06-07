# Dataset Directory

## MNIST

The MNIST dataset is **downloaded automatically** by `torchvision` the first time you run training. No manual steps are required.

After the first run the directory will look like:

```
data/
└── mnist/
    ├── MNIST/
    │   └── raw/
    │       ├── train-images-idx3-ubyte
    │       ├── train-labels-idx1-ubyte
    │       ├── t10k-images-idx3-ubyte
    │       └── t10k-labels-idx1-ubyte
    └── ...
```

## Format

- **Images**: 28 × 28 grayscale PNG / raw binary
- **Labels**: integers 0–9 representing handwritten digit classes
- **Train set**: 60 000 samples (90 % train / 10 % validation split applied at runtime)
- **Test set**: 10 000 samples

## Custom `image_folder` datasets

If you want to replace MNIST with your own data, organise it in the standard `ImageFolder` layout:

```
data/
└── my_dataset/
    ├── train/
    │   ├── class_0/
    │   │   ├── img001.png
    │   │   └── ...
    │   └── class_1/
    │       └── ...
    └── test/
        ├── class_0/
        └── class_1/
```

Then update `configs/default.yaml` → `dataset.name: image_folder` and point `dataset.data_dir` at `data/my_dataset`.
