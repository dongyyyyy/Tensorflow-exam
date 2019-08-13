import tensorflow as tf

x_data = [[3.,2.,1.],[7.,6.,5.],[5.,4.,3.]]
y_data = [[1.],[2.],[2.]]

W = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))

X = tf.placeholder(tf.float32,shape=[None,3])
Y = tf.placeholder(tf.float32,shape=[None,1])

train = tf.matmul(X,W) + b # X1*W1 + X2*W2 + X3*W3 + b

cost = tf.reduce_mean(tf.square(train-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        c_val,w_val,b_val,_ = sess.run([cost,W,b,optimizer],feed_dict={X:x_data,Y:y_data})
        if i %100 == 0:
            print("step ",i," cost : ",c_val," weight : ",w_val," bias : ",b_val)

    print("Learning finish")
    print("[10.,9.,8.] predict : ", sess.run(train,feed_dict={X:[[10.,9.,8.]]}))

