import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

image_width = 28
image_height = 28
image = np.zeros(shape=(image_height, image_width))
image_1d = np.zeros(shape=(1, image_height * image_width))
gray = 0.8
isMouseDown = False
mouseX = []
mouseY = []

def UpdateImage(x, y):
    #print("x:{0}, y:{1}".format(x, y))
    if(type(x) == None.__class__ or type(x) == None.__class__):
        return
    
    y = -y;
    image[int(y - 1)][int(x)] = image[int(y - 1)][int(x - 1)] = image[int(y - 1)][int(x + 1)] = gray    # xxx
    image[int(y)][int(x)] = image[int(y)][int(x - 1)] = image[int(y)][int(x + 1)] = gray                # xxx
    image[int(y + 1)][int(x)] = image[int(y + 1)][int(x - 1)] = image[int(y + 1)][int(x + 1)] = gray    # xxx
    
def UpdateUI(result):
    line.set_xdata(mouseX)
    line.set_ydata(mouseY)
    if result == -1:
        plt.title("")
    else:
        plt.title("{0}".format(result), fontsize = 40, color = 'red')
    plt.draw()

def OnClick(event):
    global isMouseDown
    if event.button == 1: # left
        isMouseDown = True;

def OnRelease(event):
    global image, image_1d, isMouseDown
    if event.button == 3: # right
        del mouseX[:]
        del mouseY[:]
        image = np.zeros(shape=(image_height, image_width))
        UpdateUI(-1)
    if event.button == 1: # left
        isMouseDown = False;
        recognize()
    image_1d = image.ravel()
    #print("---------------------------------------------------\n{0}\nlength = {1}".format(image, len(image)))
    #print("---------------------------------------------------\n{0}\nlength = {1}".format(image_1d, len(image_1d)))
    
def OnMotion(event):
    global isMouseDown
    if (isMouseDown):
       mouseX.append(event.xdata)
       mouseY.append(event.ydata)
       UpdateImage(event.xdata, event.ydata)
       UpdateUI(-1)

w_conv1 = tf.Variable(tf.zeros([5, 5, 1, 32]), name = "w_conv1")
b_conv1 = tf.Variable(tf.zeros([32]), name = "b_conv1")
w_conv2 = tf.Variable(tf.zeros([5, 5, 32, 64]), name = "w_conv2")
b_conv2 = tf.Variable(tf.zeros([64]), name = "b_conv2")
w_fc1 = tf.Variable(tf.zeros([7 * 7 * 64, 1024]), name = "w_fc1")
b_fc1 = tf.Variable(tf.zeros([1024]), name = "b_fc1")
w_fc2 = tf.Variable(tf.zeros([1024, 10]), name = "w_fc2")
b_fc2 = tf.Variable(tf.zeros([10]), name = "b_fc2")

def conv(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def recognize():
    global image_1d, w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'checkpoint/save')

        x = tf.constant(image_1d, dtype = tf.float32)
        x_2d = tf.reshape(x, [-1, 28, 28, 1], name = 'x_image_2d')

        h_conv1 = tf.nn.relu(conv(x_2d, w_conv1) + b_conv1)
        h_pool1 = max_pool(h_conv1)

        h_conv2 = tf.nn.relu(conv(h_pool1, w_conv2) + b_conv2)
        h_pool2 = max_pool(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)
        prediction = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    
        UpdateUI(sess.run(tf.argmax(prediction, 1)))

fig = plt.figure()
fig.canvas.mpl_connect('button_press_event', OnClick)
fig.canvas.mpl_connect('button_release_event', OnRelease)
fig.canvas.mpl_connect('motion_notify_event', OnMotion)

plt.gca().set_aspect('equal', adjustable='box')
axes = plt.gca()
axes.set_xlim(0, image_width - 1)
axes.set_ylim(-(image_height - 1), 0)
line, = axes.plot(mouseX, mouseY, 'b-', lw=20)

plt.show()
