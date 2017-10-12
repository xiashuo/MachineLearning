# _*_ coding:utf-8 _*_
from keras.datasets import mnist
from numpy import *
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import  RMSprop
(x_train, y_train), (x_test, y_test)=mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],-1)/255 #这里第2个参数为-1时，表示将剩下的维度展开成1维
x_test=x_test.reshape(x_test.shape[0],-1)/255
print(y_train)
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

model=Sequential([
    Dense(32,input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(x_train, y_train,epochs=3)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(x_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
