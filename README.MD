# Pairwise learning of MRI scans using a convolutional Siamese network for prediction of knee pain

### Prerequisites

The tool was developped based on the following dependencies:

1. PyTorch (1.1 or greater).
2. NumPy (1.16 or greater).
3. Scipy (1.30 or greater)
3. OpenCV (3.4.2 or greater)

Please note that the dependencies may require Python 3.6 or greater. It is recommended to install and maintain all packages using [`conda`](https://www.anaconda.com/). For installation of GPU accelerated PyTorch, additional effort may be required. Please check the official websites of [PyTorch](https://pytorch.org/get-started/locally/) and [CUDA](https://developer.nvidia.com/cuda-downloads) for detailed instructions.

### Data files
Original imaging file in .npy stored in:
```bash
data/raw/NAME_OF_SEQUENCE/*.npy
```
Registered imaging file in.npy stored in:
```bash
data/registered/NAME_OF_SEQUENCE/*.npy
```

Name of sequences contain
```bash
SAG_IW_TSE_LEFT & SAG_IW_TSE_RIGHT
```

## Examples
To start a fresh training with learning rate of 1e-4, batch size of 64, and learning rate decay of 0.9
```bash
$ python main.py --lr 0.0001 --bs 64 --gamma 0.9
```
To perform linear registration
```bash
$ python main.py --r
```
To continue a training from saved checkpoint:
```bash
$ python main.py --c
```

## <a name="Arguments"></a>Arguments

1 . `--rp`:
>> Run with multiple GPUs paralleling.

2 .  `--r`:
>> Run with performing linear registration.

3 . `--c`:
>> Run with the saved checkpoints.

4 . `--lr`: *float*
>> Learning rate, default is 1e-4

5 . `--gamma`: *float*

>> Learning rate decay, default is 1.0

6 . `--epochs`: *int*
>> Number of epochs, default is 500

7 . `--bs`: *int*
>> Batch size, default is 64


## Authors

* **Gary Han Chang**, garyhanchang@gmail.com - Kolachalama laboratory, Boston University School of Medicine
