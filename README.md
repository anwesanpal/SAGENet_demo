# SAGENet_demo
Demo code for SAGE-Net

## Prerequisite

1. This demo has been tested using Python3.8 with PyTorch 1.9.0 on NVIDIA GeForce RTX 2080Ti with CUDA 11.0.

2. `Theano` backend is used for most of the Keras models. To set the correct settings, modify the `~/.keras/keras.json` file as:
```
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "theano",
    "image_data_format": "channels_first"
}
```
## Steps

1. (Recommended) Create a virtual environment using virtualenv or conda:
```
virtualenv sagenet_env --python=python3.8
source sagenet_env/bin/activate
``` 

```
conda create -n sagenet_env python=3.8
conda activate sagenet_env
```

2. Install dependencies by running `pip install -r requirements.txt`

3. Clone the repository as:
```
    git clone https://github.com/anwesanpal/SAGENet_demo.git
    cd SAGENet_demo/
```

4. For each of the four algorithms, namely - `DR(eye)VE`, `BDDA`, `MLNet`, `PiCANet`, there is a directory called "demo_algorithm" folder. Inside these folders, there is a `demo.py` file with the first line specifying how to run the demo code. At the output of each `demo.py` file, two images will be created - `heatmap_bdda.jpg` and `heatmap_sage.jpg`. These images show the predicted maps as trained on the BDDA gaze vs our SAGE.