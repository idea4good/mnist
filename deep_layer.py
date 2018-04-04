import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# x for 28 * 28 images data
x = tf.placeholder(tf.float32, shape= [None, 784])
# y for probability of each digit(0-9). Such as [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = tf.placeholder(tf.float32, shape = [None, 10])

# change x from 1 dimension to 2 dimensions(28 pixels * 28 pixels)
x_2d = tf.reshape(x, [-1, 28, 28, 1], name = 'x_image_2d')

def weigth_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial, name = name)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial, name = name)

# convolution
def conv(x, w):
	return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

# pooling
def max_pool(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# 1st convolution layer -----------------------------
w_conv1 = weigth_variable([5, 5, 1, 32], 'w_conv1')
b_conv1 = bias_variable([32], 'b_conv1')

# do convolution and push through RELU activation
h_conv1 = tf.nn.relu(conv(x_2d, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

# 2nd convolution layer -----------------------------
w_conv2 = weigth_variable([5, 5, 32, 64], 'w_conv2')
b_conv2 = bias_variable([64], 'b_conv2')

# do convolution
h_conv2 = tf.nn.relu(conv(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

# full connected layer -----------------------------
w_fc1 = weigth_variable([7 * 7 * 64, 1024], 'w_fc1')
b_fc1 = bias_variable([1024], 'b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# drop out some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer -----------------------------
w_fc2 = weigth_variable([1024, 10], 'w_fc2')
b_fc2 = bias_variable([10], 'b_fc2')

# define model
prediction = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

# the loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# minimize the loss in each training
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# get accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# trainning
sess.run(tf.global_variables_initializer())
for i in range(3000):
	batch = mnist_data.train.next_batch(50)
	train_step.run(feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.5})

	# show status
	if i % 100 == 0:
		trainning_accuracy = accuracy.eval(feed_dict = {x: batch[0], y: batch[1], keep_prob: 1.0})
		print('Traning accuracy = {0:.3f}%'.format(trainning_accuracy * 100))

# save model data
saver = tf.train.Saver()
print('Save model here: {0}'.format(saver.save(sess, "checkpoint/save")))

# evaluate accuracy with test data
print ('------------------------------------\nAccuracy = {0:.3f}%'.format(accuracy.eval(feed_dict = {x: mnist_data.test.images, y: mnist_data.test.labels, keep_prob: 1.0})*100))

writer = tf.summary.FileWriter('./tensorboard-deep', sess.graph)
writer.close()
sess.close()