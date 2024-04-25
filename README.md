# Emotion detector using facial recognition

This project discusses a method, for identifying emotions based on expressions using a neural network model trained on datasets with various emotional faces.


## Train model

Python programming language was used to train the model. This language provides us with the TensorFlow 2.4.1 and Keras 2.4.3 libraries, which have the necessary tools for face and feature recognition. TensorFlow and Keras were chosen due to their popularity and ease of use in the field of deep learning. Additionally, the OpenCV library was used to detect faces in the images.

```bash
    Tensorflow = 2.4.1
    Keras = 2.4.3 
```
[Model here](https://github.com/yeison-ascanio/FaceEmotionDetection/blob/main/FaceEmotion.ipynb)


## Real time test

<img src="https://github.com/yeison-ascanio/FaceEmotionDetection/blob/main/assets/face.gif" width="500">

## Up Environment

You need install anaconda & then:

- The name FaceEmotion is optional, your environment can have any name
- h5py is the model
```bash
    $ conda create -n FaceEmotion
    $ conda activate FaceEmotion
    $ conda install python=3.7
    $ pip install tensorflow==2.4.1
    $ pip install keras==2.4.3
    $ pip install imutils opencv-python h5py
    $ pip install matplotlib == 3.2.2
```
## Run

After activating the environment, you should position yourself at the root of the project and execute the following command:
```bash
    $ python FaceEmotionVideo.py
```
