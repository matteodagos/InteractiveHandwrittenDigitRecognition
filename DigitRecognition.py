from tkinter import *
from turtle import title
from PIL import ImageGrab, Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection,metrics

global x_0, y_0
global classifier
global window,text,button,canvas

def trainNN():
    global classifier
    digits, y = load_digits(return_X_y=True)
    classifier = MLPClassifier()
    classifier.fit(digits,y)

def draw(event):
    global x_0, y_0
    global button,canvas,text
    button["state"] = ACTIVE
    text.delete(0,"end")
    if x_0 and y_0:
        canvas.create_line(x_0, y_0,x_0+1,y_0+1,x_0-1,y_0-1,
                           x_0,y_0+1,x_0,y_0-1,x_0+1,y_0,x_0-1,
                           y_0,x_0-1,y_0+1,x_0+1,y_0-1,
                           event.x, event.y,width=12,fill="black",smooth=1)
    x_0 = event.x
    y_0 = event.y

def letgo(event):
    global x_0, y_0
    global button,canvas,text
    all_segment_ids = canvas.find_all()
    all_segments = []
    for isegment in range(len(all_segment_ids)):
        all_segments.append(canvas.coords(isegment))
    x_0, y_0 = None, None
    save_canvas()

def clear_button():
    global button,canvas,text
    canvas.delete("all")
    button["state"] = DISABLED
    text.delete(0,"end")

def gui():
    global window,canvas,button,text
    global x_0,y_0
    window = Tk()
    window.title("Digit Recognition")
    canvas = Canvas(window, width=300, height=300,bg="white")
    canvas.pack()
    text = Entry(window)
    text.pack()
    button = Button(window,text="Clear",command=clear_button)
    button.pack()
    x_0 = None
    y_0 = None
    canvas.bind('<B1-Motion>', draw)
    canvas.bind('<ButtonRelease-1>', letgo)
    window.mainloop()
    
def from_screenshot_to_array(screenshot):
    X_digit = np.array(screenshot)[:,:,:3]
    X_digit = Image.fromarray(X_digit)
    X_digit = X_digit.resize((8,8))
    
    X_digit = np.array(X_digit)[:,:,:3].mean(axis = 2)
    #plt.imshow(X_digit,cmap="gray")
    #plt.show()
    X_digit = [16] - np.round((X_digit.flatten() / 255) * 16)
    text.insert(0,classifier.predict(X_digit.reshape(1,-1))[0])

def save_canvas():
    global classifier
    global window,button,canvas,text
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    xx = x + canvas.winfo_width()
    yy = y + canvas.winfo_height()
    screenshot = ImageGrab.grab(bbox=(x, y, xx, yy))
    from_screenshot_to_array(screenshot)

def main():
    trainNN()
    gui()

if __name__ == "__main__":
    main()
