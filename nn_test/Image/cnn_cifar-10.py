# _*_ coding:utf-8 _*_
from Image import cifar10_input,cifar10
import tensorflow as tf
import numpy as np
import time
import math

data_dir = 'cifar10_data/cifar-10-batches-bin'  # 下载 CIFAR-10 的默认路径
cifar10.maybe_download_and_extract() # 下载数据集，并解压、展开到其默认位置

batch_size=128
images_train,labels_train=cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
images_test,labels_test=cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)

def weight_variable(shape,stddev):
    initial=tf.truncated_normal(shape,stddev=stddev)
    return tf.Variable(initial)

def bias_variable(cons,shape):
    initial=tf.constant(cons,shape=shape)
    return tf.Variable(initial)

def conv(x,w):
    return tf.nn.conv2d(x,w,[1,1,1,1],padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

image_holder=tf.placeholder(dtype=np.float32,shape=[batch_size,24,24,3])
label_holder=tf.placeholder(dtype=tf.int32,shape=[batch_size])
#第一层
weight1=weight_variable([5,5,3,64],5e-2)
bias1=bias_variable(0.0,[64])
conv1=tf.nn.relu(conv(image_holder,weight1)+bias1)
pool1=max_pool_3x3(conv1)
#第二层
weight2=weight_variable([5,5,64,64],5e-2)
bias2=bias_variable(0.1,[64])
conv2=tf.nn.relu(conv(pool1,weight2)+bias2)
pool2=max_pool_3x3(conv2)

reshape=tf.reshape(pool2,[batch_size,-1])
dim=reshape.get_shape()[1].value
#全连接层
weight3=weight_variable([dim,384],4e-2)
bias3=bias_variable(0.1,[384])
fc3=tf.nn.relu(tf.matmul(reshape,weight3)+bias3)
#全连接层
weight4=weight_variable([384,192],4e-2)
bias4=bias_variable(0.1,[192])
fc4=tf.nn.relu(tf.matmul(fc3,weight4)+bias4)
#输出
weight5=weight_variable([192,10],1/192.0)
bias5=bias_variable(0.0,[10])
logits=tf.matmul(fc4,weight5)+bias5

def loss(logits,labels):
    labels=tf.cast(labels,tf.int64)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy_per_example')
    cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')

loss=loss(logits,label_holder)
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op=tf.nn.in_top_k(logits,label_holder,1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

def train():
    for step in range(3000):
        start_time=time.time()
        image_batch,label_batch=sess.run([images_train,labels_train])
        _,loss_value=sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})
        duration=time.time()-start_time
        if step % 50==0:
            exmples_per_sec=batch_size/duration
            sec_per_batch=float(duration)
            print("step %s,loss=%s,(%s example/sec,%s sec/batch)"%(step,loss_value,exmples_per_sec,sec_per_batch))

def test():
    num_examples=10000
    num_iter=int(math.ceil(num_examples/batch_size))
    true_count=0
    total_count=num_iter*batch_size
    step=0
    while step<num_iter:
        image_batch,label_batch=sess.run([images_test,labels_test])
        predictions=sess.run(top_k_op,feed_dict={image_holder:image_batch,label_holder:label_batch})
        true_count+=np.sum(predictions)
        step+=1
    precision=true_count/total_count
    print("precision=",precision)

if __name__=="__main__":
    train()
    test()











