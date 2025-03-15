import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageFilter, ImageEnhance
from skimage.filters import gaussian
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os
import math

class EntopticVisuals:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Entoptic Visuals")
        
        # Variables for sliders
        self.scale_var = tk.DoubleVar(value=10)
        self.speed_var = tk.DoubleVar(value=0.1)
        self.intensity_var = tk.DoubleVar(value=10)
        self.halftone_var = tk.DoubleVar(value=5)
        
        # New control variables
        self.pattern_type_var = tk.StringVar(value="random")
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.dot_size_var = tk.DoubleVar(value=1.0)
        
        # State variables
        self.running = False
        self.frame_index = 0
        self.source_type = "entoptic"  # 'entoptic', 'image', or 'video'
        self.source_data = None
        self.video_capture = None
        self.video_frames = []
        self.current_size = (400, 400)  # Default size
        
        # Initialize UI components
        self.setup_ui()
        
    def setup_ui(self):
        # Frame for the display
        display_frame = ttk.Frame(self.root)
        display_frame.pack(pady=10)
        
        # Set up matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.axis('off')
        self.img_display = self.ax.imshow(self.generate_entoptic_frame(0), cmap='gray')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack()
        
        # Control buttons frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5)
        
        # Source selection buttons
        ttk.Button(control_frame, text="Use Entoptic Pattern", command=self.use_entoptic).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        
        # Animation controls
        animation_frame = ttk.Frame(self.root)
        animation_frame.pack(pady=5)
        ttk.Button(animation_frame, text="Start Animation", command=self.start_animation).pack(side=tk.LEFT, padx=5)
        ttk.Button(animation_frame, text="Stop Animation", command=self.stop_animation).pack(side=tk.LEFT, padx=5)
        ttk.Button(animation_frame, text="Reset", command=self.reset_animation).pack(side=tk.LEFT, padx=5)
        
        # Pattern type dropdown
        pattern_frame = ttk.Frame(self.root)
        pattern_frame.pack(pady=5)
        ttk.Label(pattern_frame, text="Pattern Type:").pack(side=tk.LEFT, padx=5)
        pattern_types = ["random", "circles", "lines", "squares", "diamonds"]
        pattern_dropdown = ttk.Combobox(pattern_frame, textvariable=self.pattern_type_var, values=pattern_types, width=10)
        pattern_dropdown.pack(side=tk.LEFT, padx=5)
        pattern_dropdown.bind("<<ComboboxSelected>>", self.on_slider_change)
        
        # Sliders
        self.setup_sliders()
    
    def setup_sliders(self):
        # Original controls
        controls = [
            ("Scale", self.scale_var, 1, 20),
            ("Speed", self.speed_var, 0.01, 1),
            ("Intensity", self.intensity_var, 1, 30),
            ("Halftone Size", self.halftone_var, 1, 10)
        ]
        
        # New controls
        new_controls = [
            ("Dot Size", self.dot_size_var, 0.5, 3),
            ("Contrast", self.contrast_var, 0.5, 2),
            ("Brightness", self.brightness_var, 0.5, 2)
        ]
        
        # Add original controls
        for label, var, min_val, max_val in controls:
            self.add_slider(label, var, min_val, max_val)
            
        # Add separator
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill='x', pady=10)
        
        # Add new controls
        for label, var, min_val, max_val in new_controls:
            self.add_slider(label, var, min_val, max_val)
    
    def add_slider(self, label, var, min_val, max_val):
        frame = ttk.Frame(self.root)
        frame.pack(pady=2)
        ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT, padx=5)
        scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL, command=self.on_slider_change)
        scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        scale.config(length=250)
        value_label = ttk.Label(frame, text=f"{var.get():.2f}")
        value_label.pack(side=tk.LEFT, padx=5)
        var.trace_add("write", lambda *args, v=var, lbl=value_label: self.update_label(v, lbl))
    
    def update_label(self, var, label):
        label.config(text=f"{var.get():.2f}")
    
    def on_slider_change(self, event=None):
        self.update_display()
    
    def generate_entoptic_frame(self, i, size=None):
        if size is None:
            size = self.current_size
            
        scale = int(self.scale_var.get())
        speed = self.speed_var.get()
        intensity = self.intensity_var.get()
        
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
        
        # Apply contrast and brightness
        img = self.apply_contrast_brightness(img)
        
        img_array = np.array(img.convert('L')) / 255.0
        
        # Apply halftone effect
        halftone_img = self.apply_halftone(img_array)
        return halftone_img
    
    def apply_contrast_brightness(self, img):
        # Apply contrast
        contrast = self.contrast_var.get()
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        # Apply brightness
        brightness = self.brightness_var.get()
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
            
        return img
    
    def apply_halftone(self, img_array):
        halftone_size = int(self.halftone_var.get())
        dot_size = self.dot_size_var.get()
        pattern_type = self.pattern_type_var.get()
        
        # Apply gaussian blur based on halftone size
        img_array = gaussian(img_array, sigma=halftone_size)
        
        h, w = img_array.shape
        result = np.zeros((h, w), dtype=np.uint8)
        
        if pattern_type == "random":
            # Original random dots pattern
            result = (img_array > np.random.rand(*img_array.shape)).astype(np.uint8) * 255
            
        elif pattern_type == "circles":
            # Circle pattern
            cell_size = int(10 * dot_size)
            for y in range(0, h, cell_size):
                for x in range(0, w, cell_size):
                    # Define the region
                    y_end = min(y + cell_size, h)
                    x_end = min(x + cell_size, w)
                    
                    # Get the average intensity in this cell
                    if y < h and x < w:
                        avg_intensity = np.mean(img_array[y:y_end, x:x_end])
                        
                        # Calculate the circle radius based on intensity
                        radius = int((1 - avg_intensity) * cell_size * 0.5 * dot_size)
                        
                        if radius > 0:
                            # Draw circle
                            cy, cx = (y + y_end) // 2, (x + x_end) // 2
                            y_indices, x_indices = np.ogrid[-cy:h-cy, -cx:w-cx]
                            mask = x_indices**2 + y_indices**2 <= radius**2
                            result[mask] = 255
            
        elif pattern_type == "lines":
            # Lines pattern
            line_thickness = max(1, int(3 * dot_size))
            line_spacing = int(10 * dot_size)
            
            for y in range(0, h, line_spacing):
                for x in range(0, w):
                    # Calculate intensity at this position
                    intensity = img_array[min(y, h-1), min(x, w-1)]
                    # Calculate line height based on intensity
                    line_height = int((1 - intensity) * line_spacing * dot_size)
                    
                    # Draw line
                    line_start = y
                    line_end = min(y + line_height, h)
                    if x < w and line_end > line_start:
                        result[line_start:line_end, x] = 255
        
        elif pattern_type == "squares":
            # Square pattern
            cell_size = int(10 * dot_size)
            for y in range(0, h, cell_size):
                for x in range(0, w, cell_size):
                    # Define the region
                    y_end = min(y + cell_size, h)
                    x_end = min(x + cell_size, w)
                    
                    # Get the average intensity in this cell
                    if y < h and x < w:
                        avg_intensity = np.mean(img_array[y:y_end, x:x_end])
                        
                        # Calculate the square size based on intensity
                        size = int((1 - avg_intensity) * cell_size * dot_size)
                        half_size = size // 2
                        
                        # Draw square
                        cy, cx = (y + y_end) // 2, (x + x_end) // 2
                        y_start = max(0, cy - half_size)
                        y_end = min(h, cy + half_size)
                        x_start = max(0, cx - half_size)
                        x_end = min(w, cx + half_size)
                        
                        if y_end > y_start and x_end > x_start:
                            result[y_start:y_end, x_start:x_end] = 255
        
        elif pattern_type == "diamonds":
            # Diamond pattern
            cell_size = int(10 * dot_size)
            for y in range(0, h, cell_size):
                for x in range(0, w, cell_size):
                    # Define the region
                    y_end = min(y + cell_size, h)
                    x_end = min(x + cell_size, w)
                    
                    # Get the average intensity in this cell
                    if y < h and x < w:
                        avg_intensity = np.mean(img_array[y:y_end, x:x_end])
                        
                        # Calculate the diamond size based on intensity
                        size = int((1 - avg_intensity) * cell_size * dot_size)
                        
                        if size > 0:
                            # Draw diamond
                            cy, cx = (y + y_end) // 2, (x + x_end) // 2
                            
                            # Create diamond shape
                            for dy in range(-size, size+1):
                                diamond_width = size - abs(dy)
                                y_pos = cy + dy
                                if 0 <= y_pos < h:
                                    for dx in range(-diamond_width, diamond_width+1):
                                        x_pos = cx + dx
                                        if 0 <= x_pos < w:
                                            result[y_pos, x_pos] = 255
        
        return result
    
    def process_current_frame(self):
        if self.source_type == "entoptic":
            return self.generate_entoptic_frame(self.frame_index)
        
        elif self.source_type == "image" and self.source_data is not None:
            # Apply contrast and brightness adjustments
            img = self.apply_contrast_brightness(self.source_data.copy())
            img_array = np.array(img.convert('L')) / 255.0
            return self.apply_halftone(img_array)
        
        elif self.source_type == "video" and self.video_frames:
            frame_idx = self.frame_index % len(self.video_frames)
            frame = self.video_frames[frame_idx]
            
            # Convert to PIL Image for contrast/brightness
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_img = self.apply_contrast_brightness(pil_img)
            
            # Convert back to grayscale numpy array
            gray_frame = np.array(pil_img.convert('L')) / 255.0
            return self.apply_halftone(gray_frame)
        
        # Default fallback
        return self.generate_entoptic_frame(0)
    
    def update_display(self):
        current_frame = self.process_current_frame()
        self.img_display.set_array(current_frame)
        self.canvas.draw()
    def update_animation(self):
        if self.running:
            self.frame_index += 1
            self.update_display()
            self.root.after(50, self.update_animation)
    
    def start_animation(self):
        self.running = True
        self.update_animation()
    
    def stop_animation(self):
        self.running = False
    
    def reset_animation(self):
        self.frame_index = 0
        self.update_display()
    
    def use_entoptic(self):
        self.source_type = "entoptic"
        self.frame_index = 0
        self.update_display()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            try:
                image = Image.open(file_path)
                # Resize to maintain reasonable dimensions
                image = self.resize_image(image)
                self.current_size = image.size
                self.source_data = image
                self.source_type = "image"
                self.frame_index = 0
                self.update_display()
            except Exception as e:
                print(f"Error loading image: {e}")
    
    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            try:
                # Release previous video if any
                if self.video_capture is not None:
                    self.video_capture.release()
                
                self.video_capture = cv2.VideoCapture(file_path)
                self.video_frames = []
                
                # Extract frames
                max_frames = 100  # Limit number of frames to avoid memory issues
                count = 0
                
                while count < max_frames:
                    ret, frame = self.video_capture.read()
                    if not ret:
                        break
                    
                    # Resize frame to maintain reasonable dimensions
                    height, width = frame.shape[:2]
                    max_dimension = 400
                    if height > max_dimension or width > max_dimension:
                        if height > width:
                            new_height = max_dimension
                            new_width = int(width * (max_dimension / height))
                        else:
                            new_width = max_dimension
                            new_height = int(height * (max_dimension / width))
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    self.video_frames.append(frame)
                    count += 1
                    # Only read every few frames to reduce memory usage for longer videos
                    if count % 3 == 0 and self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT) > 100:
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.video_capture.get(cv2.CAP_PROP_POS_FRAMES) + 2)
                
                if self.video_frames:
                    self.current_size = (self.video_frames[0].shape[1], self.video_frames[0].shape[0])
                    self.source_type = "video"
                    self.frame_index = 0
                    self.update_display()
                    print(f"Loaded {len(self.video_frames)} frames from video")
                else:
                    print("No frames could be extracted from the video")
                
            except Exception as e:
                print(f"Error loading video: {e}")
    
    def resize_image(self, image):
        width, height = image.size
        max_dimension = 400
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            return image.resize((new_width, new_height), Image.LANCZOS)
        return image

if __name__ == "__main__":
    root = tk.Tk()
    app = EntopticVisuals(root)
    root.mainloop()
