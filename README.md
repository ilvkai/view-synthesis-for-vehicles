# Pose-based View Synthesis for Vehicles: a Perspective Aware Method
This repo is based on pose-gan [https://github.com/AliaksandrSiarohin/pose-gan], which is an excellent work for person pose transfer.
### Requirment
* python2.7
* Numpy
* Scipy
* Skimage
* Pandas
* Tensorflow
* Keras
* [keras-contrib](https://github.com/keras-team/keras-contrib)
* tqdm 

### Environments
1. pip install -r requirements.txt
2. install keras-contrib by running: pip install git+https://www.github.com/keras-team/keras-contrib.git

### Training
In order to train a model:
1. Download VeRi dataset https://github.com/VehicleReId/VeRidataset. Put it in data folder. 
Rename this folder to data/VeRi-dataset.
Rename image_train and image_test with test and train. 

2. Download [view-synthesis-for-vehicles.zip](https://drive.google.com/file/d/1qDaCTft9zLuojMlKyEydW8OBdIn_9RXq/view?usp=sharing) 
and unzip the contents to the main folder. 

3. Training the full system by running the train.sh:
bash train.sh

### Testing

bash test.sh



