import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import datetime

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential

def plot_examples(x_train):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(x_train[i])
        plt.title(y_train[i])

    plt.show()


def create_model():
    resnet = keras.applications.resnet50.ResNet50(
        include_top=False,
        input_shape=(32, 32, 3),
        weights=None,
        classes=10
    )
    x = layers.Flatten()(resnet.output)
    x = layers.Dense(10, activation='softmax')(x)
    return keras.models.Model(resnet.input, x)


def preprocess_data(X, y):
    X = keras.applications.resnet50.preprocess_input(X)
    y = keras.utils.to_categorical(y)
    return X, y


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

model = create_model()
opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(x_train[0])
print(y_train[0])
print(model.summary())

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1
)

epochs = 50

model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=128,
    epochs=epochs,
    verbose=1,
    callbacks=[tensorboard_callback]
)

model.save(f'cifar10_e_{epochs}')







