import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x for 28 * 28 images data
x = tf.placeholder(tf.float32, shape=[None, 784])
# y for probability of each digit(0-9). Such as [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = tf.placeholder(tf.float32, [None, 10])

# define modle
weight = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))
y_predic = tf.nn.softmax(tf.matmul(x, weight) + bias)

# the loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predic))

# minimize the loss in each training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# training
for i in range(1000):
	batch_xs, batch_ys = mnist_data.train.next_batch(100) # get 100 random data
	sess.run(train_step, feed_dict={x: batch_xs, y:batch_ys}) # optimize the loss with data
	#print("weight = ", sess.run(weight), "bias = ", sess.run(bias)) # track training process

cross_prediction = tf.equal(tf.argmax(y_predic, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

# test the result
test_accuracy = sess.run(accuracy, feed_dict={x: mnist_data.test.images, y: mnist_data.test.labels})
print("Test accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()