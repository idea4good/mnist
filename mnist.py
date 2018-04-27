import os
import os.path
import shutil, math
import tensorflow as tf

LOGDIR = "logs/"
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)

def conv_layer(input, weight, bias):
  conv = tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding="SAME")
  act = tf.nn.relu(conv + bias)
  ret =  tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

  tf.summary.histogram("weights", weight)
  tf.summary.histogram("biases", bias)
  tf.summary.histogram("activations", act)
  for i in range(weight.shape[3]):
      tf.summary.image("conv_weight", tf.reshape(weight[:, :, 0, i], [-1, weight.shape[0], weight.shape[1], 1]), 1)
  for i in range(ret.shape[3]):
      tf.summary.image("conv_output", tf.reshape(ret[0, :, :, i], [-1, ret.shape[1], ret.shape[2], 1]), 1)
  return ret

def fc_layer(input, weight, bias):
  act = tf.matmul(input, weight) + bias
  tf.summary.histogram("weights", weight)
  tf.summary.histogram("biases", bias)
  tf.summary.histogram("activations", act)
  return act

def build_model(x, y):
  with tf.name_scope("input"):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('image', x_image, 1)

  with tf.name_scope("conv"):
    w = tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[16]), name="B")
    conv_out = conv_layer(x_image, w, b)

  with tf.name_scope("fc1"):
    w = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[1024]), name="B")
    fc1 = fc_layer(tf.reshape(conv_out, [-1, 7 * 7 * 64]), w, b)
    relu = tf.nn.relu(fc1)
    tf.summary.histogram("relu", relu)

  with tf.name_scope("fc2"):
    w = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[10]), name="B")
    logits = fc_layer(relu, w, b)
    tf.summary.histogram("predict", tf.argmax(logits[0], 0))

  with tf.name_scope("loss"):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="loss")
    tf.summary.scalar("loss", loss)

  with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1E-3).minimize(loss)

  with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

  return (train_step, accuracy)

if __name__ == '__main__':
  tf.reset_default_graph()
  sess = tf.Session()

  x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
  y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
  (train_step, accuracy) = build_model(x, y)

  summary_op = tf.summary.merge_all()
  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(LOGDIR)
  writer.add_graph(sess.graph)
  
  for i in range(2001):
    batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
    if i % 50 == 0:
      [train_accuracy, s] = sess.run([accuracy, summary_op], feed_dict={x: batch[0], y: batch[1]})
      writer.add_summary(s, i)
      print("Iteration: %s, Accuracy:%s"%(i, train_accuracy))

  saver = tf.train.Saver()
  saver.save(sess, LOGDIR + "model.ckpt")

  print('Training done!')
  print('Run `tensorboard --logdir=%s` for more details.' % LOGDIR)
