from keras.datasets import mnist
from keras import layers, models
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import joblib
import ssl

# Crear un contexto SSL que no verifique los certificados
ssl._create_default_https_context = ssl._create_unverified_context


(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

x_train = train_data.reshape((60000, 28*28))
x_train = x_train.astype('float32')/255

x_test = test_data.reshape((10000, 28*28))
x_test = x_test.astype('float32')/255

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=7, batch_size=128)

model.evaluate(x_test, y_test)
model.save('number_reconizer.keras')
# joblib.dump(model, 'number_reconizer.h5')