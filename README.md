# Emotion detector using facial recognition

This project discusses a method, for identifying emotions based on expressions using a neural network model trained on datasets with various emotional faces.

> [!WARNING] You'll need permissions camera for run this project
> [!WARNING] This project was developed in macOs

## Train model

Python programming language was used to train the model. This language provides us with the TensorFlow 2.4.1 and Keras 2.4.3 libraries, which have the necessary tools for face and feature recognition. TensorFlow and Keras were chosen due to their popularity and ease of use in the field of deep learning. Additionally, the OpenCV library was used to detect faces in the images.


```bash
    Tensorflow
    Keras
```
[Model here](https://github.com/yeison-ascanio/FaceEmotionDetection/blob/main/FaceEmotion.ipynb)


## Real time test

<img src="https://github.com/yeison-ascanio/FaceEmotionDetection/blob/main/assets/face.gif" width="500">

## Up Environment

> [!NOTE] You can use 2 methods for get up the environment. 1st native python, 2st with anaconda

> [!NOTE] Native python
Create the environment:
```python
    python3 -m venv /route/environment
    python -m venv /route/environment
```
> [!NOTE] if your system is windows, maybe it change

activate the environment:
```python
    source /route/environment/bin/activate
````
> [!TIP] for windows:
```python
    \route\environment\Scripts\activate
````
Now, to install dependencies, run the following command:
```python
    pip install -r requirements.txt
```

> [!NOTE] Anaconda method
```python
    conda create -n FaceEmotion
    conda activate FaceEmotion
    conda install python=3.7
    pip install tensorflow==2.4.1
    pip install keras==2.4.3
    pip install imutils opencv-python h5py
    pip install matplotlib == 3.2.2
```
## Run

After activating the environment, you should position yourself at the root of the project and execute the following command:
```python
    python FaceEmotionVideo.py
```
