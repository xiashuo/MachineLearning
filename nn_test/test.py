# _*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()  # 替换成这样就好
sess=tf.Session()
sess.run(init)
for step in range(201):
    sess.run(train)
w=sess.run(Weights)
b=sess.run(biases)
fig=plt.figure()
ax=fig.add_subplot(111)
y=w*x_data+b
ax.plot(x_data,y)
plt.show()