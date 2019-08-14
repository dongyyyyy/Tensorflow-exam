import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

classes = 10

#Using batch_norm

def model(X,mode):
    conv1 = tf.layers.conv2d(inputs=X,filters=32,kernel_size=3,strides=1,padding='SAME') # 28 28 32
    bn1 = tf.layers.batch_normalization(conv1,training=mode)
    relu1 = tf.nn.relu(bn1)
    maxpool1 = tf.layers.max_pooling2d(inputs=relu1,pool_size=2,strides=2,padding='SAME')

    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=64, kernel_size=[3, 3], strides=1, padding='SAME') # 7
    bn2 = tf.layers.batch_normalization(conv2,training=mode)
    relu2 = tf.nn.relu(bn2)
    maxpool2 = tf.layers.max_pooling2d(inputs=relu2,pool_size=2,strides=2,padding='SAME')

    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=128, kernel_size=[3, 3], strides=1, padding='SAME') # 4
    bn3 = tf.layers.batch_normalization(conv3,training=mode)
    relu3 = tf.nn.relu(bn3)
    maxpool3 = tf.layers.max_pooling2d(inputs=relu3, pool_size=2, strides=2, padding='SAME')

    avgpool = tf.layers.average_pooling2d(inputs=maxpool3,pool_size= 4, strides= 4) # 1 1 128
    flat = tf.reshape(avgpool,[-1,128])
    out = tf.layers.dense(inputs=flat,units=classes)
    return out

def main():
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for
    # more information about the mnist dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X = tf.placeholder(tf.float32, [None, 784])
    X_img = tf.reshape(X, [-1, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, classes])
    mode = tf.placeholder(tf.bool)

    train = model(X_img,mode)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=train,labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    correct = tf.equal(tf.argmax(train, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    num_epoch = 15
    batch_size = 100
    num_iterations = int(mnist.train.num_examples / batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epoch):
            avg_cost = 0

            for i in range(num_iterations):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys,mode:True})
                avg_cost += cost_val / num_iterations

            print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

        print("Learning finished")

        print(
            "Accuracy: ",
            accuracy.eval(
                session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels, mode:False}
            ),
        )

        # Get one and predict
        r = random.randint(0, mnist.test.num_examples - 1)
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[r: r + 1], 1)),mnist.test.labels[r: r + 1] )
        print(
            "Prediction: ",
            sess.run(tf.argmax(train, 1), feed_dict={X: mnist.test.images[r: r + 1],mode:False}),sess.run(train, feed_dict={X: mnist.test.images[r: r + 1],mode:False})
        )

        plt.imshow(
            mnist.test.images[r: r + 1].reshape(28, 28),
            cmap="Greys",
            interpolation="nearest",
        )
        plt.show()

if __name__ == '__main__':
    main()