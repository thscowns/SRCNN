import tensorflow as tf
from Data import data_generate

x_train,y_train = data_generate()

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])

w1 = tf.Variable(tf.random_normal([2,2]))
b1 = tf.Variable(tf.random_normal([2,1]))
w2 = tf.Variable(tf.random_normal([2,1]))
b2 = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(tf.sigmoid(tf.matmul(X,w1)),w2) + b2)

cost = - tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        x_train,y_train = data_generate()
        for step in range(5000):
            cost_val , _ = sess.run([cost,train],feed_dict ={X : x_train, Y : y_train})
            #if step % 1000 == 0:
             #   print(step,cost_val)

        h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X: x_train,Y : y_train})
        print(i,"result : ",a)