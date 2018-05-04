import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches
import mnist, math

image_width = 28
image_height = 28
image = np.zeros(shape=(image_height, image_width))
image_1d = np.zeros(shape=(1, image_height * image_width))
gray = 0.9
isMouseDown = False
sess = tf.InteractiveSession()

def UpdateImage(x, y):
    if(type(x) == None.__class__ or type(x) == None.__class__):
        return
    x = int(round(x))
    y = int(round(y))
    image_y = -(y + 1)
    if (image_y - 2) < 0 or image_y >= image_height or (x + 2) >= image_width:
        return

    axes.add_patch(patches.Rectangle((x , y), 3, 3))
    plt.draw()

    image[image_y][x] = image[image_y][x + 1] = image[image_y][x + 2] = gray
    image[image_y - 1][x] = image[image_y - 1][x + 1] = image[image_y - 1][x + 2] = gray    
    image[image_y - 2 ][x] = image[image_y - 2][x + 1] = image[image_y - 2][x + 2] = gray
    
def update_figure(result):
    if result == -1:
        axes.set_xlabel("")
    else:
        axes.set_xlabel("Predict: {0}".format(result))
    plt.draw()

def OnClick(event):
    global isMouseDown
    if event.button == 1: # left
        isMouseDown = True;
        UpdateImage(event.xdata, event.ydata)

def OnRelease(event):
    global image, image_1d, isMouseDown
    if event.button == 3: # right
        image = np.zeros(shape=(image_height, image_width))
        reset_axis(axes)
        update_figure(-1)
    if event.button == 1: # left
        isMouseDown = False;
        recognize()
    image_1d = image.ravel()

def OnMotion(event):
    global isMouseDown
    if (isMouseDown):
       UpdateImage(event.xdata, event.ydata)
       update_figure(-1)

def recognize():
    x = tf.constant(image_1d, dtype = tf.float32)
    x_2d = tf.reshape(x, [-1, 28, 28, 1], name = 'x_image_2d')

    conv_out1 = mnist.conv_layer(x_2d, w_conv1, b_conv1)
    conv_out2 = mnist.conv_layer(conv_out1, w_conv2, b_conv2)
    features = conv_out2.get_shape()[1:4].num_elements()
    flatten_conv = tf.reshape(conv_out2, [-1, features])

    fc1_out = mnist.fc_layer(flatten_conv, w_fc1, b_fc1)
    relu = tf.nn.relu(fc1_out)
    logits = mnist.fc_layer(relu, w_fc2, b_fc2)

    update_figure(sess.run(tf.argmax(logits, 1)))
    plot_conv_cout(conv_out1, 2, 'conv layer 1')
    plot_conv_cout(conv_out2, 3, 'conv layer 2')

def reset_axis(axes):
    plt.cla()
    axes.set_xlim(0, image_width)
    axes.xaxis.set_major_locator(MultipleLocator(4))
    axes.xaxis.set_minor_locator(MultipleLocator(1))
    axes.xaxis.grid(True, which='minor')

    axes.set_ylim(-image_height, 0)
    axes.yaxis.set_major_locator(MultipleLocator(4))
    axes.yaxis.set_minor_locator(MultipleLocator(1))
    axes.yaxis.grid(True, which='minor')

def plot_conv_cout(values, index, title):
    values = sess.run(values)
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))

    _, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='gray')
        ax.set_xticks([]); ax.set_yticks([])
    plt.figure(index).canvas.set_window_title(title)
    plt.show()

with tf.name_scope("conv1"):
    w_conv1 = tf.Variable(tf.zeros([5, 5, 1, 16]), name = "W")
    b_conv1 = tf.Variable(tf.zeros([16]), name = "B")
with tf.name_scope("conv2"):
    w_conv2 = tf.Variable(tf.zeros([5, 5, 16, 36]), name = "W")
    b_conv2 = tf.Variable(tf.zeros([36]), name = "B")
with tf.name_scope("fc1"):
    w_fc1 = tf.Variable(tf.zeros([7 * 7 * 36, 1024]), name = "W")
    b_fc1 = tf.Variable(tf.zeros([1024]), name = "B")
with tf.name_scope("fc2"):
    w_fc2 = tf.Variable(tf.zeros([1024, 10]), name = "W")
    b_fc2 = tf.Variable(tf.zeros([10]), name = "B")

saver = tf.train.Saver()
saver.restore(sess, mnist.LOGDIR + "model.ckpt")

fig = plt.figure('Input')
fig.canvas.mpl_connect('button_press_event', OnClick)
fig.canvas.mpl_connect('button_release_event', OnRelease)
fig.canvas.mpl_connect('motion_notify_event', OnMotion)

plt.gca().set_aspect('equal', adjustable='box')
axes = plt.gca()
reset_axis(axes)
plt.show()
