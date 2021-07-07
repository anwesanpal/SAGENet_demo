# SAGENet_demo
Demo code for our CVPR 2020 paper **"Looking at the Right Stuff" - Guided Semantic-Gaze for Autonomous Driving**. Training code will be uploaded soon.

## Prerequisite

This demo has been tested using Python3.8 with PyTorch 1.9.0 on NVIDIA GeForce RTX 2080Ti with CUDA 11.0.

## Steps

1. (Recommended) Create a virtual environment using conda:
```
conda create -n sagenet_env python=3.8
conda activate sagenet_env
```

2. Clone the repository as:
```
git clone https://github.com/anwesanpal/SAGENet_demo.git
cd SAGENet_demo/
```

3. Install dependencies by running `pip install -r requirements.txt`.

4. `Theano` backend is used for most of the Keras models. To set the correct settings, modify the `~/.keras/keras.json` file as:
```
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "theano",
    "image_dim_ordering": "th",
    "image_data_format": "channels_first"
}```

5. Download the pretrained models:
```
gdown https://drive.google.com/uc?id=1jpY2uyj1jk3qLiDDvI7UuDgfgTjHUj8e
unzip pretrained_models.zip && rm -rf pretrained_models.zip
```

6. For each of the four algorithms, namely - `DR(eye)VE`, `BDDA`, `MLNet`, `PiCANet`, there is a directory called "demo_algorithm" folder. Inside these folders, there is a `demo.py` file with the first line specifying the command line to run the demo code. At the output of each `demo.py` file, two images will be created - `heatmap_bdda.jpg` and `heatmap_sage.jpg`. These images show the predicted maps as trained on the BDDA gaze vs our SAGE.

7. Additionally, inside `demo_dreyeve`, there is a `predict_dreyeve_video.py` script which takes as input a video from the `videos` folder and runs detection on it. To get the videos, follow the instructions to download and save the videos in the top-level directory:
```
gdown https://drive.google.com/uc?id=1u1fyG9ZAHNVZuNXYSk81ANqOeWO5O4u9
unzip videos.zip && rm -rf videos.zip
```
The script uses ffmpeg to save the output videos, so that needs to be installed as well:
```
sudo apt update
sudo apt install ffmpeg
```
Run the script by following the command line instructions. The output of this script is a video showing the detections on the input video. Change the argument `--gt_type` to `bdda`, or `sage` to get BDDA gaze and SAGE respectively.

## Citations

Please cite our work if you found this research useful for your work:

```bash
@InProceedings{Pal_2020_CVPR,
author = {Pal, Anwesan and Mondal, Sayan and Christensen, Henrik I.},
title = {"Looking at the Right Stuff" - Guided Semantic-Gaze for Autonomous Driving},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
} 
```