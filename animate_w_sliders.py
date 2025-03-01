import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageFilter
from skimage.filters import gaussian
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()
root.title("Entoptic Visuals")

scale_var = tk.DoubleVar(value=10)
speed_var = tk.DoubleVar(value=0.1)
intensity_var = tk.DoubleVar(value=10)
halftone_var = tk.DoubleVar(value=5)

running = False  # Control de la animación

def generate_entoptic_frame(i, size=(120 ,120)):
    scale = int(scale_var.get())
    speed = speed_var.get()
    intensity = intensity_var.get()
    halftone_size = int(halftone_var.get())
    
    x = np.linspace(-3, 3, size[0] // scale)
    y = np.linspace(-3, 3, size[1] // scale)
    X, Y = np.meshgrid(x, y)
    
    Z1 = np.sin(X * intensity + i * speed) * np.cos(Y * intensity - i * speed)
    Z2 = np.sin((X**2 + Y**2) * intensity + i * speed)
    pattern = (Z1 + Z2) / 2
    
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255
    pattern = pattern.astype(np.uint8)
    
    img = Image.fromarray(pattern)
    img = img.resize(size, resample=Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    
    img_array = np.array(img.convert('L')) / 255.0
    img_array = gaussian(img_array, sigma=halftone_size)
    img_array = (img_array > np.random.rand(*img_array.shape)).astype(np.uint8) * 255
    
    return img_array

fig, ax = plt.subplots()
ax.axis('off')
img_display = ax.imshow(generate_entoptic_frame(0), cmap='gray')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

def update_animation():
    if running:
        img_display.set_array(generate_entoptic_frame(update_animation.frame))
        canvas.draw()
        update_animation.frame += 1
        root.after(50, update_animation)
update_animation.frame = 0

def start_animation():
    global running
    running = True
    update_animation()

def stop_animation():
    global running
    running = False

def on_slider_change(event=None):
    img_display.set_array(generate_entoptic_frame(0))
    canvas.draw()

def update_label(var, label):
    label.config(text=f"{var.get():.2f}")

controls = [
    ("Scale", scale_var, 1, 20),
    ("Speed", speed_var, 0.01, 1),
    ("Intensity", intensity_var, 1, 30),
    ("Halftone Size", halftone_var, 1, 10)
]

for label, var, min_val, max_val in controls:
    frame = ttk.Frame(root)
    frame.pack()
    ttk.Label(frame, text=label).pack(side=tk.LEFT)
    scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL, command=on_slider_change)
    scale.pack(side=tk.LEFT)
    value_label = ttk.Label(frame, text=f"{var.get():.2f}")
    value_label.pack(side=tk.LEFT)
    var.trace_add("write", lambda *args, v=var, lbl=value_label: update_label(v, lbl))

ttk.Button(root, text="Start Animation", command=start_animation).pack()
ttk.Button(root, text="Stop Animation", command=stop_animation).pack()

root.mainloop()

