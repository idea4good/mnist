import numpy as np
import matplotlib.pyplot as plt
import model_data as model

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

def recognize():
    global image_1d
    result = np.dot(image_1d, model.weight) + model.bias
    UpdateUI(np.argmax(result))
       
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
