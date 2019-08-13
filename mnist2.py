import tensorflow as tf
import random
import matplotlib.pyplot as plt

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

classes = 10

X = tf.placeholder(tf.float32,shape=[None,784]) # 784 -> reshape 28 X 28
Y = tf.placeholder(tf.float32,shape=[None,classes])
X_img = tf.reshape(X,[-1,28,28,1]) # 28 X 28 X 1 ( Grey )
keep_prob = tf.placeholder(tf.float32)


conv1 = tf.layers.conv2d(inputs=X_img,filters=32,kernel_size=[3,3],strides=2,padding='SAME') # 14 X 14 X 32
#activation function relu
print(conv1.shape)
relu1 = tf.nn.relu(conv1)
dropout1 = tf.nn.dropout(x=relu1,keep_prob=keep_prob) # 얼만큼의 비율의 노드를 학습할때 사용할 것인가

conv2 = tf.layers.conv2d(inputs=dropout1,filters=64,kernel_size=[3,3],strides=2,padding='SAME') # 7 X 7 X 64
#activation function relu
print(conv2.shape)
relu2 = tf.nn.relu(conv2)
dropout2 = tf.nn.dropout(x=relu2,keep_prob=keep_prob) # 얼만큼의 비율의 노드를 학습할때 사용할 것인가

conv3 = tf.layers.conv2d(inputs=dropout2,filters=128,kernel_size=[3,3],strides=2,padding='SAME') # 4 X 4 X 128
#activation function relu
print(conv3.shape)
relu3 = tf.nn.relu(conv3)
dropout3 = tf.nn.dropout(x=relu3,keep_prob=keep_prob) # 얼만큼의 비율의 노드를 학습할때 사용할 것인가
# 4 X 4 X 128

avg_pool = tf.layers.average_pooling2d(inputs=dropout3,pool_size=4,strides=4) # 1 X 1 X 128
print(avg_pool.shape)
flat = tf.reshape(avg_pool,[-1,128]) # 128
print(flat.shape)

hypothesis = tf.layers.dense(inputs=flat,units=classes) # 10

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

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
            c_val,_ = sess.run([cost,optimizer],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.7})
            avg_cost += c_val/num_interations
        print("Epoch ",epoch," avg cost : ",avg_cost)

    print("Learning finish")

    print("Accuracy : ",accuracy.eval(session=sess,feed_dict={X:mnist.test.images,Y:mnist.test.labels,keep_prob:1.0}))

    r = random.randint(0,mnist.test.num_examples-1)# 0 ~ 9999
    print("Label: ",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("Predict : ", sess.run(tf.argmax(hypothesis,1),feed_dict={X:mnist.test.images[r:r+1],keep_prob:1.0}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap="Greys",interpolation="nearest") # 784 -> 28 X 28
    plt.show()
