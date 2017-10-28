# _*_ coding:utf-8 _*_
import tensorflow as tf
#输入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("minst_data",one_hot=True)
#定义网络的超参数
learning_rate=1e-3
training_iters=2e5
batch_size=128
display_step=10
#定义网络的参数
n_input=784
n_classes=10
#输入占位符
x=tf.placeholder(dtype=tf.float32,shape=[None,n_input])
y=tf.placeholder(dtype=tf.float32,shape=[None,n_classes])
keep_prob=tf.placeholder(dtype=tf.float32)

#构建模型
#定义卷积操作
def conv2d(name,x,w,b,strides=1):
    x=tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding="SAME")
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x,name=name)

#定义池化层操作
def max_pool(name,x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding="SAME",name=name)

#规范化操作
def norm(name,l_input,lsize=4):
    return tf.nn.lrn(l_input,lsize,bias=1.0,alpha=1e-3/9.0,beta=0.75,name=name)

#定义所有的网络参数
weights={
    'wc1':tf.Variable(tf.random_normal([11,11,1,96])),
    'wc2':tf.Variable(tf.random_normal([5,5,96,256])),
    'wc3':tf.Variable(tf.random_normal([3,3,256,384])),
    'wc4':tf.Variable(tf.random_normal([3,3,384,384])),
    'wc5':tf.Variable(tf.random_normal([3,3,384,256])),
    'w_fc1':tf.Variable(tf.random_normal([1*1*256,4096])),
    'w_fc2':tf.Variable(tf.random_normal([4096,4096])),
    'out':tf.Variable(tf.random_normal([4096,10]))
}
biases={
    'bc1':tf.Variable(tf.random_normal([96])),
    'bc2':tf.Variable(tf.random_normal([256])),
    'bc3':tf.Variable(tf.random_normal([384])),
    'bc4':tf.Variable(tf.random_normal([384])),
    'bc5':tf.Variable(tf.random_normal([256])),
    'b_fc1':tf.Variable(tf.random_normal([4096])),
    'b_fc2':tf.Variable(tf.random_normal([4096])),
    'out':tf.Variable(tf.random_normal([10]))
}
#定义整个网络
def alex_net(x,weights,biases,dropout):
    x=tf.reshape(x,shape=[-1,28,28,1])
    #第一层卷积
    conv1=conv2d('conv1',x,weights['wc1'],biases['bc1'])
    #下采样
    pool1=max_pool('pool1',conv1)
    #规范化
    norma1=norm('norm1',pool1)

    #第二层卷积
    conv2=conv2d('conv2',norma1,weights['wc2'],biases['bc2'])
    pool2=max_pool('pool2',conv2)
    norma2=norm('norm2',pool2)

    #第三层卷积
    conv3=conv2d('conv3',norma2,weights['wc3'],biases['bc3'])
    pool3=max_pool('pool3',conv3)
    norma3=norm('norm3',pool3)

    #第四层卷积
    conv4=conv2d('conv4',norma3,weights['wc4'],biases['bc4'])
    pool4=max_pool('pool4',conv4)
    norma4=norm('norm4',pool4)

    #第五层卷积
    conv5=conv2d('conv5',norma4,weights['wc5'],biases['bc5'])
    pool5=max_pool('pool5',conv5)
    norm5=norm('norm5',pool5)

    #全连接层1
    fc1=tf.reshape(norm5,shape=[-1,weights['w_fc1'].get_shape().as_list()[0]])
    fc1=tf.add(tf.matmul(fc1,weights['w_fc1']),biases['b_fc1'])
    fc1=tf.nn.relu(fc1)
    #dropout
    fc1=tf.nn.dropout(fc1,keep_prob=dropout)

    #全连接层2
    # fc2=tf.reshape(fc1,shape=[-1,weights['w_fc2'].get_shape().as_list()[0]])
    fc2=tf.add(tf.matmul(fc1,weights['w_fc2']),biases['b_fc2'])
    fc2=tf.nn.relu(fc2)
    fc2=tf.nn.dropout(fc2,dropout)

    #输入层
    out=tf.add(tf.matmul(fc2,weights['out']),biases['out'])
    return out

#构建模型
pred=alex_net(x,weights,biases,keep_prob)
#定义损失函数和优化器
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
#评估函数
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step=1
    while step*batch_size<training_iters:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:0.75})
        if step % display_step==0:
            #计算损失和精确度
            loss,acc=sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
            print("iter:"+str(step*batch_size)+",loss= %s,training accuracy=%s"%(loss,acc))
        step+=1
    print("finished!")
    #计算测试集的准确度
    print("testing accuracy: ",sess.run(accuracy,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob:1.0}))








