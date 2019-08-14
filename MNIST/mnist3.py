import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

classes = 10

#Using model function

def model(X, keep_prob):
    print(X.shape)
    conv1 = tf.layers.conv2d(inputs=X,filters=32,kernel_size=3,strides=1,padding='SAME') # 28 28 32
    print(conv1.shape)
    relu1 = tf.nn.relu(conv1)
    maxpool1 = tf.layers.max_pooling2d(inputs=relu1,pool_size=2,strides=2,padding='SAME')
    print(maxpool1.shape)
    dropout1 = tf.nn.dropout(x=maxpool1,keep_prob=keep_prob)

    conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], strides=1, padding='SAME') # 7
    print(conv2.shape)
    relu2 = tf.nn.relu(conv2)
    maxpool2 = tf.layers.max_pooling2d(inputs=relu2,pool_size=2,strides=2,padding='SAME')
    print(maxpool2.shape)
    dropout2 = tf.nn.dropout(x=maxpool2,keep_prob=keep_prob)

    conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], strides=1, padding='SAME') # 4
    print(conv3.shape)
    relu3 = tf.nn.relu(conv3)
    maxpool3 = tf.layers.max_pooling2d(inputs=relu3, pool_size=2, strides=2, padding='SAME')
    print(maxpool3.shape)
    dropout3 = tf.nn.dropout(x=maxpool3, keep_prob=keep_prob)

    avgpool = tf.layers.average_pooling2d(inputs=dropout3,pool_size= 4, strides= 4) # 1 1 128
    print(avgpool.shape)
    flat = tf.reshape(avgpool,[-1,128])
    print(flat.shape)
    out = tf.layers.dense(inputs=flat,units=classes)
    print(out.shape)
    return out

def main():
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for
    # more information about the mnist dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X = tf.placeholder(tf.float32, [None, 784])
    X_img = tf.reshape(X, [-1, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, classes])
    keep_prob = tf.placeholder(tf.float32)

    train = model(X_img,keep_prob)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=train,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    correct = tf.equal(tf.argmax(train, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    num_epoch = 30
    batch_size = 100
    num_iterations = int(mnist.train.num_examples / batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epoch):
            avg_cost = 0

            for i in range(num_iterations):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys,keep_prob:0.7})
                avg_cost += cost_val / num_iterations

            print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

        print("Learning finished")

        print(
            "Accuracy: ",
            accuracy.eval(
                session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob:1.0}
            ),
        )

        # Get one and predict
        r = random.randint(0, mnist.test.num_examples - 1)
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[r: r + 1], 1)))
        print(
            "Prediction: ",
            sess.run(tf.argmax(train, 1), feed_dict={X: mnist.test.images[r: r + 1],keep_prob:1.0}),
        )

        plt.imshow(
            mnist.test.images[r: r + 1].reshape(28, 28),
            cmap="Greys",
            interpolation="nearest",
        )
        plt.show()

main()
