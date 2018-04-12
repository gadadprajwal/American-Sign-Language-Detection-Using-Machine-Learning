from keras.models import load_model
import cv2
import numpy as np
import math
import os
from keras.preprocessing.image import img_to_array, load_img
#from PIL import Image
import keras
import scipy.misc
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import Conv2D,MaxPooling2D
#from keras.models import save_model
import h5py
test_path = ['/home/prajwal/PycharmProjects/Project/test_dataset']
paths = ['/home/prajwal/PycharmProjects/Project/asl_dataset']
TOTAL_DATASET = 2515
x_train = []  # training lists
y_train = []
x_test = []  # test lists
y_test = []

nb_classes = 36  # number of classes
img_rows, img_cols = 400, 400  # size of training images
img_channels = 3  # BGR channels
batch_size = 32
nb_epoch = 100  # iterations for training
data_augmentation = True

# dictionary for classes from char to numbers
classes = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'a': 10,
    'b': 11,
    'c': 12,
    'd': 13,
    'e': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'j': 19,
    'k': 20,
    'l': 21,
    'm': 22,
    'n': 23,
    'o': 24,
    'p': 25,
    'q': 26,
    'r': 27,
    's': 28,
    't': 29,
    'u': 30,
    'v': 31,
    'w': 32,
    'x': 33,
    'y': 34,
    'z': 35,
}

def load_data_set():
    for path in paths:
        for root, directories, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".jpeg"):
                    fullpath = os.path.join(root, filename)
                    img = load_img(fullpath)

                    img = scipy.misc.imresize(img, 1 / 8)
                    img = img_to_array(img)
                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('gray_image', gray_image)

                    x_train.append(np.reshape(gray_image,(50,50,1)))
                    t = fullpath.rindex('/')
                    fullpath = fullpath[0:t]
                    n = fullpath.rindex('/')
                    y_train.append(classes[fullpath[n + 1:t]])
    return x_train,y_train

def load_test_data_set():
    for path in test_path:
        for root, directories, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".jpeg"):
                    fullpath = os.path.join(root, filename)
                    img = load_img(fullpath)

                    img = scipy.misc.imresize(img, 1 / 8)
                    img = img_to_array(img)
                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('gray_image', gray_image)

                    x_test.append(np.reshape(gray_image,(50,50,1)))
                    t = fullpath.rindex('/')
                    fullpath = fullpath[0:t]
                    n = fullpath.rindex('/')
                    y_test.append(classes[fullpath[n + 1:t]])
    return x_test,y_test


def onehot(labels):
    label2=[]
    for i in labels:
        fea = [0] * 36
        fea[i]=1
        label2.append(fea)
    return np.array(label2)

def probability_prediction(pred):
    k = np.argmax(pred)
    prediction = [0] * 36
    prediction[k] = 1
    return prediction

def predict_accuracy(xtrain,ytrain,xtest,ytest,model):
    accuracy = 0
    for img,label in zip(xtest,ytest):
        prediction = model.predict(np.reshape(img, (1, 50, 50, 1)))
        k = np.argmax(prediction)
        if(k==label):
            accuracy +=1
    return accuracy/len(ytest)

def main():
    x,y = load_data_set()
    x_np = np.array(x)
    y_np = np.array(y)
    y_np_onehot = onehot(y_np)
    print(x_np.shape,y_np_onehot.shape)

    #testing data
    x_t, y_t = load_test_data_set()
    x_t_np = np.array(x_t)
    y_t_np = np.array(y_t)


    #model = make_model(x_np.shape)
    #Saving the model
    #model.save('my_model_weights.h5')
    #Loading the model
    model = load_model('my_model_weights.h5')

    '''model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])'''

    '''model.fit(x_np, y_np_onehot,
              batch_size=50,
              epochs=10,
              verbose=1,
              validation_data=None,
              callbacks=None)'''

    accuracy = predict_accuracy(x_np,y_np,x_t,y_t,model)

    '''img = cv2.imread("hand5_z_dif_seg_5_cropped.jpeg")
    img = scipy.misc.imresize(img, (50,50))
    img = img_to_array(img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prediction = model.predict(np.reshape(gray_image,(1,50,50,1)))
    prediction = probability_prediction(prediction)
    print("Prediction",prediction)'''
    print(y_t_np.shape)
    print(accuracy)



if __name__ == '__main__':
    main()