
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageFilter
from skimage.filters import gaussian

def generate_entoptic_frame(i, size=(800, 800), scale=10, speed=0.1, intensity=10, halftone_size=5):
    x = np.linspace(-3, 3, size[0] // scale)
    y = np.linspace(-3, 3, size[1] // scale)
    X, Y = np.meshgrid(x, y)
    
    # Animación con desplazamiento en el tiempo
    Z1 = np.sin(X * intensity + i * speed) * np.cos(Y * intensity - i * speed)
    Z2 = np.sin((X**2 + Y**2) * intensity + i * speed)
    pattern = (Z1 + Z2) / 2
    
    # Normalizar valores
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255
    pattern = pattern.astype(np.uint8)
    
    # Convertir a imagen y aplicar desenfoque
    img = Image.fromarray(pattern)
    img = img.resize(size, resample=Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Convertir a numpy array para aplicar halftone
    img_array = np.array(img.convert('L')) / 255.0
    img_array = gaussian(img_array, sigma=halftone_size)  # Suavizado para efecto halftone
    img_array = (img_array > np.random.rand(*img_array.shape)).astype(np.uint8) * 255  # Dither halftone
    
    return img_array

def animate_entoptic(frames=10, size=(800, 800), scale=100, speed=0.001, intensity=5, halftone_size=2):
    fig, ax = plt.subplots()
    ax.axis('off')
    img_display = ax.imshow(generate_entoptic_frame(0, size, scale, speed, intensity, halftone_size), cmap='gray')
    
    def update(frame):
        img_display.set_array(generate_entoptic_frame(frame, size, scale, speed, intensity, halftone_size))
        return [img_display]
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    plt.show()
    
# Ejecutar animación
animate_entoptic()

