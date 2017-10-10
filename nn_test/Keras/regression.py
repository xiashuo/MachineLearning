# _*_ coding:utf-8 _*_
from keras.models import Sequential
from keras.layers import Dense,Activation
from numpy import *
from keras import backend as K
import matplotlib.pyplot as plt
random.seed(1337)  # for reproducibility

X=linspace(-1,1,200)
random.shuffle(X)#打乱顺序
model=Sequential()
Y=0.5*X+2+random.normal(0,0.05,[200])#random.normal()正太分布函数
plt.scatter(X,Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]     # first 160 data points
X_test, Y_test = X[160:], Y[160:]       # last 40 data points

model=Sequential()
model.add(Dense(units=1,input_dim=1))
model.compile(loss='mse', optimizer='sgd')

print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()