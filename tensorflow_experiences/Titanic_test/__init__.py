# _*_ coding:utf-8 _*_
import pandas as pd
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import numpy as np

data=pd.read_csv("data/train.csv")
#取部分特征字段用于分类，并将所有缺失的字段填充为0
data['Sex']=data['Sex'].apply(lambda s:1 if s=='male' else 0)
data=data.fillna(0)
dataset_X=data[['Sex','Age','Pclass','SibSp','Parch','Fare']]
dataset_X=dataset_X.as_matrix()
#两种分类分别为幸存和死亡，增加一个字段‘Deceased’，表示死亡，取值为‘、Survived’字段取非
data['Deceased']=data['Survived'].apply(lambda s:int(not s))
dataset_Y=data[['Deceased','Survived']]
dataset_Y=dataset_Y.as_matrix()
#使用sklearn的train_test_split函数将标记数据切分成“训练集和验证集”
X_train,X_test,Y_train,Y_test=train_test_split(dataset_X,dataset_Y,test_size=0.2,random_state=42)

#构建计算图
with tf.name_scope("input"):
    x=tf.placeholder(tf.float32,[None,6])
    y=tf.placeholder(tf.float32,[None,2])
with tf.name_scope("classifier"):
    w=tf.Variable(tf.random_normal([6,2]),name='weights')
    b=tf.Variable(tf.zeros([2]),name='bias')
    y_pred = tf.nn.softmax(tf.matmul(x, w) + b)
    tf.summary.histogram("weights",w)
    tf.summary.histogram("bias",b)

saver=tf.train.Saver()
#代价函数使用交叉熵
with tf.name_scope("cost"):
    cross_entropy=-tf.reduce_sum(y*tf.log(y_pred+1e-10),axis=1)
    #批量样本的代价值为所有样本交叉熵的平均值
    cost=tf.reduce_mean(cross_entropy)
    tf.summary.scalar("loss",cost)

#使用随机梯度下降优化算法
train_op=tf.train.GradientDescentOptimizer(0.001).minimize(cost)

#精确度
with tf.name_scope("accuracy"):
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y_test, 1))
    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar("accuracy",acc_op)
#训练
def train():
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs", sess.graph)
        merged = tf.summary.merge_all()
        tf.global_variables_initializer().run()
        for epotch in range(10):
            total_loss=0.0
            for i in range(len(X_train)):
                _,loss=sess.run([train_op,cost],feed_dict={x:[X_train[i]],y:[Y_train[i]]})
                total_loss+=loss
            print("epotch:%4d,total_loss=%.9f"%(epotch,total_loss))
            summary, accuracy = sess.run([merged, acc_op], feed_dict={x: X_test, y: Y_test})
            writer.add_summary(summary, epotch)
        print("Training complete!")
        save_path=saver.save(sess,"save/model.ckpt")
        # correct=np.equal(np.argmax(pred,1),np.argmax(Y_test,1))
        # accuracy=np.mean(correct.astype(np.float32))
        print("目前精度为：",accuracy)

#验证
def test():
    with tf.Session() as sess:
        saver.restore(sess,"save/model.ckpt")
        testdata=pd.read_csv("data/test.csv").fillna(0)
        testdata['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
        X_test=testdata[['Sex','Age','Pclass','SibSp','Parch','Fare']]
        predictions=np.argmax(sess.run(y_pred,feed_dict={x:X_test}),1)
        #构建提交结果的数据结构，并将结果存储为csv文件
        submission=pd.DataFrame({
            "PassengerId":testdata["PassengerId"],
            "Survived":predictions
        })
        submission.to_csv("titanic-submit.csv",index=False)


if __name__=="__main__":
    train()






