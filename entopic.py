import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def generate_entoptic_image(size=(800, 800), scale=10):
    x = np.linspace(-3, 3, size[0] // scale)
    y = np.linspace(-3, 3, size[1] // scale)
    X, Y = np.meshgrid(x, y)
    
    # Crear patrones geométricos con funciones trigonométricas
    Z1 = np.sin(X * 10) * np.cos(Y * 10)
    Z2 = np.sin((X**2 + Y**2) * 5)
    pattern = (Z1 + Z2) / 2
    
    # Normalizar los valores para que estén entre 0 y 255
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255
    pattern = pattern.astype(np.uint8)
    
    # Convertir a imagen
    img = Image.fromarray(pattern)
    img = img.resize(size, resample=Image.BILINEAR)
    
    # Aplicar efecto de desenfoque para darle un look más orgánico
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Agregar color mediante un colormap de Matplotlib
    cmap = plt.get_cmap('twilight')
    colored_img = cmap(np.array(img) / 255.0)
    colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
    
    # Guardar y mostrar la imagen
    final_img = Image.fromarray(colored_img)
    final_img.show()
    
# Generar la imagen
generate_entoptic_image()

