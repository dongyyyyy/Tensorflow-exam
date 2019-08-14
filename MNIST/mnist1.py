import tensorflow as tf
import random
import matplotlib.pyplot as plt

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

classes = 10

X = tf.placeholder(tf.float32,shape=[None,784]) # 784 -> reshape 28 X 28
Y = tf.placeholder(tf.float32,shape=[None,classes])

W = tf.Variable(tf.random_normal([784,classes]))
b = tf.Variable(tf.random_normal([classes]))

hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

correct = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

batch_size = 100 # 한번에 100개의 이미지를 가져와서 학습을 시키겠다.
num_epoch = 15 # 총 60000장의 이미지를 15번 학습하겠다.
num_interations = int(mnist.train.num_examples / batch_size)
# 한번에 100개의 이미지를 학습하기 때문에 총 60000장의 이미지를 600번 나눠서 반복해서 학습함을 의미한다

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epoch):
        avg_cost = 0
        for i in range(num_interations):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            c_val,_ = sess.run([cost,optimizer],feed_dict={X:batch_x,Y:batch_y})
            avg_cost += c_val/num_interations
        print("Epoch ",epoch," avg cost : ",avg_cost)

    print("Learning finish")

    print("Accuracy : ",accuracy.eval(session=sess,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))

    r = random.randint(0,mnist.test.num_examples-1)# 0 ~ 9999
    print("Label: ",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("Predict : ", sess.run(tf.argmax(hypothesis,1),feed_dict={X:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap="Greys",interpolation="nearest") # 784 -> 28 X 28
    plt.show()
