import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches
import model_data as model

image_width = 28
image_height = 28
image = np.zeros(shape=(image_height, image_width))
image_1d = np.zeros(shape=(1, image_height * image_width))
gray = 0.9
isMouseDown = False

def UpdateImage(x, y):
    #print("x:{0}, y:{1}".format(x, y))
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
    
def UpdateUI(result):
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
        image = np.zeros(shape=(image_height, image_width))
        reset_axis(axes)
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
       UpdateImage(event.xdata, event.ydata)
       UpdateUI(-1)

def recognize():
    global image_1d
    result = np.dot(image_1d, model.weight) + model.bias
    UpdateUI(np.argmax(result))

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

fig = plt.figure()
fig.canvas.mpl_connect('button_press_event', OnClick)
fig.canvas.mpl_connect('button_release_event', OnRelease)
fig.canvas.mpl_connect('motion_notify_event', OnMotion)

plt.gca().set_aspect('equal', adjustable='box')
axes = plt.gca()
reset_axis(axes)
plt.show()
