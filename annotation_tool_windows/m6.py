# Part 1/5: Imports, Global Variables, and Basic Utilities

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
import datetime
import threading
import time
from typing import List, Tuple, Optional, Dict, Any
from scipy.signal import savgol_filter

# Global variables for timestamp directory
CURRENT_TIMESTAMPED_DIR = None
TIMESTAMPED_DIRNAME = None

# Global variable to track labeled keyframes (for UI panel updates)
LABELED_KEYFRAMES = set()


# Part 2/5: UI Components & Utilities

class ProgressDialog:
    """Dialog to show loading progress with animation"""
    def __init__(self, parent, title="Loading..."):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.configure(bg='#333333')
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Prevent closing with the X button
        self.dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        
        style = ttk.Style(self.dialog)
        style.configure("Loading.Horizontal.TProgressbar", 
                        background='#4CAF50', 
                        troughcolor='#555555')
        
        self.main_frame = ttk.Frame(self.dialog, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_label = ttk.Label(self.main_frame, text="Loading frames...", font=("Arial", 12))
        self.status_label.pack(pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(self.main_frame, orient="horizontal", length=300,
                                            mode="determinate", style="Loading.Horizontal.TProgressbar")
        self.progress_bar.pack(pady=10, fill=tk.X)
        
        self.frame_count_label = ttk.Label(self.main_frame, text="Frames: 0", font=("Arial", 10))
        self.frame_count_label.pack(pady=5)
        
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
    def update_progress(self, value, maximum, frame_count=None):
        self.progress_bar["maximum"] = maximum
        self.progress_bar["value"] = value
        percentage = int((value / maximum) * 100) if maximum > 0 else 0
        self.status_label.config(text=f"Loading frames... {percentage}%")
        if frame_count is not None:
            self.frame_count_label.config(text=f"Frames loaded: {frame_count}")
        self.dialog.update_idletasks()
        self.dialog.update()
        
    def update(self):
        self.dialog.update_idletasks()
        self.dialog.update()
        
    def close(self):
        self.dialog.grab_release()
        self.dialog.destroy()


class ImageCanvas(ttk.Frame):
    """
    Custom canvas class that handles image display, zooming, panning, and drawing.
    Also saves the previous mask state to allow one-step undo.
    """
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drawing = False
        self.last_x, self.last_y = 0, 0
        self.image_loaded = False
        self.loading_indicator = None
        self.canvas = tk.Canvas(self, bg='#222222', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.h_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        self.debug_mode = False
        self._setup_bindings()
        
    def _setup_bindings(self):
        self.canvas.bind("<Button-1>", self._start_drawing)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_drawing)
        self.canvas.bind("<MouseWheel>", self._zoom_with_mousewheel)
        self.canvas.bind("<Button-4>", self._zoom_in)
        self.canvas.bind("<Button-5>", self._zoom_out)
        self.canvas.bind("<Button-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-2>", self._stop_pan)
        self.canvas.bind("<Button-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-3>", self._stop_pan)
        self.canvas.bind("<Control-d>", self._toggle_debug)
        
    def _toggle_debug(self, event=None):
        self.debug_mode = not self.debug_mode
        self.app.update_display()
        
    def _start_drawing(self, event):
        if not self.app.has_frames():
            return
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        # Save current mask state for one-step undo
        self.app.previous_mask = self.app.current_mask().copy()
        self._draw(event)
        
    def _draw(self, event):
        if not self.drawing or not self.app.has_frames():
            return
        original_h, original_w = self.app.current_frame().shape[:2]
        x, y = self._canvas_to_image_coords(event.x, event.y)
        if 0 <= x < original_w and 0 <= y < original_h:
            cv2.circle(self.app.current_mask(), (x, y), self.app.brush_size,
                       255 if self.app.tool == "draw" else 0, -1)
            self.app.update_display()
            
    def _stop_drawing(self, event):
        self.drawing = False
        
    def _start_pan(self, event):
        self.canvas.config(cursor="fleur")
        self.last_x = event.x
        self.last_y = event.y
        
    def _pan(self, event):
        if not self.app.has_frames():
            return
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.pan_x += dx
        self.pan_y += dy
        self.last_x = event.x
        self.last_y = event.y
        self.app.update_display()
        
    def _stop_pan(self, event):
        self.canvas.config(cursor="")
        
    def _zoom_with_mousewheel(self, event):
        if event.delta > 0:
            self._zoom_in(event)
        else:
            self._zoom_out(event)
            
    def _zoom_in(self, event):
        if not self.app.has_frames():
            return
        old_img_x, old_img_y = self._canvas_to_image_coords(event.x, event.y)
        self.zoom_factor = min(self.zoom_factor * 1.2, 10.0)
        self._update_zoom_pan(event.x, event.y, old_img_x, old_img_y)
        
    def _zoom_out(self, event):
        if not self.app.has_frames():
            return
        old_img_x, old_img_y = self._canvas_to_image_coords(event.x, event.y)
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
        self._update_zoom_pan(event.x, event.y, old_img_x, old_img_y)
        
    def _update_zoom_pan(self, mouse_x, mouse_y, old_img_x, old_img_y):
        if not self.app.has_frames():
            return
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        original_h, original_w = self.app.current_frame().shape[:2]
        new_width = int(original_w * self.zoom_factor)
        new_height = int(original_h * self.zoom_factor)
        center_x = (canvas_width - new_width) / 2
        center_y = (canvas_height - new_height) / 2
        new_mouse_x = center_x + old_img_x * self.zoom_factor + self.pan_x
        new_mouse_y = center_y + old_img_y * self.zoom_factor + self.pan_y
        self.pan_x += (mouse_x - new_mouse_x)
        self.pan_y += (mouse_y - new_mouse_y)
        self.app.update_display()
        
    def reset_view(self):
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.app.update_display()
        
    def center_image(self):
        if not self.app.has_frames():
            return
        self.pan_x = 0
        self.pan_y = 0
        self.app.update_display()
        
    def show_loading_indicator(self):
        self.clear_canvas()
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2 - 20,
            text="Loading image...",
            fill="white",
            font=("Arial", 14)
        )
        bar_width = 200
        bar_height = 10
        x0 = (self.canvas.winfo_width() - bar_width) // 2
        y0 = (self.canvas.winfo_height() - bar_height) // 2 + 20
        self.canvas.create_rectangle(x0, y0, x0 + bar_width, y0 + bar_height,
                                     outline="white", width=1)
        self.loading_indicator = self.canvas.create_rectangle(
            x0 + 2, y0 + 2, x0 + 2, y0 + bar_height - 2,
            fill="#4CAF50", outline=""
        )
        self._animate_loading(0, bar_width, x0)
        
    def _animate_loading(self, step, bar_width, x0):
        if self.loading_indicator:
            progress = step / 100 if step <= 100 else (200 - step) / 100
            progress_width = int(bar_width * progress)
            y0 = (self.canvas.winfo_height() - 10) // 2 + 20
            self.canvas.coords(self.loading_indicator,
                               x0 + 2, y0 + 2, x0 + 2 + progress_width, y0 + 10 - 2)
            if not self.image_loaded:
                self.canvas.after(20, lambda: self._animate_loading((step + 2) % 200, bar_width, x0))
        
    def hide_loading_indicator(self):
        self.image_loaded = True
        self.loading_indicator = None
        
    def clear_canvas(self):
        self.canvas.delete("all")
        
    def _canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> Tuple[int, int]:
        if not self.app.has_frames():
            return 0, 0
        original_h, original_w = self.app.current_frame().shape[:2]
        display_info = self.app.display_info
        if not display_info:
            return 0, 0
        img_x, img_y, img_w, img_h = display_info
        if (img_x <= canvas_x <= img_x + img_w and img_y <= canvas_y <= img_y + img_h):
            rel_x = (canvas_x - img_x) / img_w
            rel_y = (canvas_y - img_y) / img_h
            image_x = int(rel_x * original_w)
            image_y = int(rel_y * original_h)
            return image_x, image_y
        return 0, 0


class FrameNavigator(ttk.Frame):
    """
    Manages frame navigation with slider, skip frames, and buttons.
    """
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.slider_updating = False
        self.nav_frame = ttk.Frame(self)
        self.nav_frame.pack(fill=tk.X, pady=5)
        self.prev_btn = ttk.Button(self.nav_frame, text="Previous", command=self._prev_frame)
        self.prev_btn.pack(side=tk.LEFT, padx=2)
        self.next_btn = ttk.Button(self.nav_frame, text="Next", command=self._next_frame)
        self.next_btn.pack(side=tk.LEFT, padx=2)
        self.frame_label = ttk.Label(self.nav_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.nav_frame, text="Skip:").pack(side=tk.LEFT, padx=5)
        self.skip_var = tk.IntVar(value=50)
        self.skip_entry = ttk.Spinbox(self.nav_frame, from_=1, to=1000, width=5, textvariable=self.skip_var)
        self.skip_entry.pack(side=tk.LEFT, padx=2)
        self.slider_frame = ttk.Frame(self)
        self.slider_frame.pack(fill=tk.X, pady=2)
        self.frame_slider = ttk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self._slider_changed)
        self.frame_slider.pack(fill=tk.X, padx=10, expand=True)
        self.frame_slider.bind("<ButtonRelease-1>", self._slider_released)
        
    def _prev_frame(self):
        self.app.prev_frame()
        
    def _next_frame(self):
        self.app.next_frame(self.skip_var.get())
        
    def _slider_changed(self, value):
        if self.slider_updating or not self.app.has_frames():
            return
        max_idx = len(self.app.frames) - 1
        if max_idx <= 0:
            return
        frame_idx = int(float(value) / 100 * max_idx)
        self.app.set_current_frame(frame_idx)
    
    def _slider_released(self, event):
        if not self.app.has_frames():
            return
        value = self.frame_slider.get()
        max_idx = len(self.app.frames) - 1
        if max_idx <= 0:
            return
        frame_idx = int(float(value) / 100 * max_idx)
        self.app.set_current_frame(frame_idx)
        
    def update_slider_position(self, current_idx, total_frames):
        if total_frames <= 1:
            self.frame_slider.set(0)
            return
        self.slider_updating = True
        position = (current_idx / (total_frames - 1)) * 100
        self.frame_slider.set(position)
        self.slider_updating = False
        
    def update_frame_label(self, current_idx, total_frames):
        self.frame_label.config(text=f"Frame: {current_idx + 1}/{total_frames}")


class ToolbarManager(ttk.Frame):
    """
    Manages the application toolbar UI elements.
    """
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.tool_frame = ttk.Frame(self)
        self.tool_frame.pack(side=tk.LEFT, padx=5)
        self.draw_btn = ttk.Button(self.tool_frame, text="Draw", width=10,
                                     command=lambda: self.app.set_tool("draw"))
        self.draw_btn.pack(side=tk.LEFT, padx=2)
        self.erase_btn = ttk.Button(self.tool_frame, text="Erase", width=10,
                                      command=lambda: self.app.set_tool("erase"))
        self.erase_btn.pack(side=tk.LEFT, padx=2)
        self.tool_indicator = ttk.Label(self.tool_frame, text="Active: Draw", width=15)
        self.tool_indicator.pack(side=tk.LEFT, padx=10)
        ttk.Label(self, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        self.brush_scale = ttk.Scale(self, from_=1, to=20, orient=tk.HORIZONTAL, command=self.app.update_brush_size)
        self.brush_scale.set(5)
        self.brush_scale.pack(side=tk.LEFT, padx=5)
        # Add Undo button
        self.undo_btn = ttk.Button(self, text="Undo", command=self.app.undo_last_action, width=10)
        self.undo_btn.pack(side=tk.LEFT, padx=5)
        # Zoom controls on right side
        zoom_frame = ttk.Frame(self)
        zoom_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Button(zoom_frame, text="Zoom +", width=8, command=lambda: self.app.zoom_in()).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Zoom -", width=8, command=lambda: self.app.zoom_out()).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset View", width=10, command=lambda: self.app.reset_view()).pack(side=tk.LEFT, padx=2)

    def update_tool_indicator(self, tool):
        self.tool_indicator.config(text=f"Active: {tool.capitalize()}")



class ImageEnhancement(ttk.LabelFrame):
    """
    Manages image enhancement filters and settings.
    """
    def __init__(self, parent, app):
        super().__init__(parent, text="Image Enhancement")
        self.app = app
        self.brightness_var = tk.IntVar(value=0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.blur_var = tk.IntVar(value=0)
        self.kalman_var = tk.BooleanVar(value=False)
        # Add new variables for low pass and high pass filters
        self.low_pass_var = tk.IntVar(value=0)  # Strength of low pass filter (0-100)
        self.high_pass_var = tk.IntVar(value=0)  # Strength of high pass filter (0-100)
        self.prev_filtered_frame = None
        self._setup_ui()
        
    def _setup_ui(self):
        # Create two columns for the filters to better organize the UI
        left_col = ttk.Frame(self)
        left_col.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        right_col = ttk.Frame(self)
        right_col.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Left column controls
        ttk.Label(left_col, text="Brightness:").grid(row=0, column=0, sticky="w", padx=5)
        self.brightness_scale = ttk.Scale(left_col, from_=-100, to=100, variable=self.brightness_var,
                                          command=lambda x: self.app.update_display())
        self.brightness_scale.grid(row=0, column=1, sticky="ew", padx=5)
        self.brightness_scale.configure(length=150)
        
        ttk.Label(left_col, text="Contrast:").grid(row=1, column=0, sticky="w", padx=5)
        self.contrast_scale = ttk.Scale(left_col, from_=0.5, to=2.0, variable=self.contrast_var,
                                        command=lambda x: self.app.update_display())
        self.contrast_scale.grid(row=1, column=1, sticky="ew", padx=5)
        self.contrast_scale.configure(length=150)
        
        ttk.Label(left_col, text="Gaussian Blur:").grid(row=2, column=0, sticky="w", padx=5)
        self.blur_scale = ttk.Scale(left_col, from_=0, to=15, variable=self.blur_var,
                                    command=lambda x: self.app.update_display())
        self.blur_scale.grid(row=2, column=1, sticky="ew", padx=5)
        self.blur_scale.configure(length=150)
        
        # Right column controls
        ttk.Label(right_col, text="Low Pass Filter:").grid(row=0, column=0, sticky="w", padx=5)
        self.low_pass_scale = ttk.Scale(right_col, from_=0, to=100, variable=self.low_pass_var,
                                      command=lambda x: self.app.update_display())
        self.low_pass_scale.grid(row=0, column=1, sticky="ew", padx=5)
        self.low_pass_scale.configure(length=150)
        
        ttk.Label(right_col, text="High Pass Filter:").grid(row=1, column=0, sticky="w", padx=5)
        self.high_pass_scale = ttk.Scale(right_col, from_=0, to=100, variable=self.high_pass_var,
                                       command=lambda x: self.app.update_display())
        self.high_pass_scale.grid(row=1, column=1, sticky="ew", padx=5)
        self.high_pass_scale.configure(length=150)
        
        # Kalman smoothing checkbox (moved to right column, row 2)
        self.kalman_check = ttk.Checkbutton(right_col, text="Temporal Smoothing (Naive Kalman)",
                                            variable=self.kalman_var, command=self.reset_kalman)
        self.kalman_check.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Make columns expandable
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        left_col.columnconfigure(1, weight=1)
        right_col.columnconfigure(1, weight=1)
        
    def apply_low_pass_filter(self, frame, strength):
        """Apply a low pass filter (blur) to the image with given strength"""
        if strength <= 0:
            return frame
        
        # Convert strength (0-100) to kernel size (must be odd)
        kernel_size = int(1 + (strength / 10)) * 2 + 1
        
        # Apply Gaussian blur as low pass filter
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    def apply_high_pass_filter(self, frame, strength):
        """Apply a high pass filter to enhance edges"""
        if strength <= 0:
            return frame
        
        # Apply low pass filter to get low frequencies
        low_pass = self.apply_low_pass_filter(frame, strength)
        
        # High pass = original - low pass (weighted by strength)
        weight = strength / 100.0
        high_pass = cv2.addWeighted(frame, 1 + weight, low_pass, -weight, 0)
        
        return high_pass
        
    def apply_filters(self, frame):
        if frame is None:
            return None
            
        f = frame.astype(np.float32)
        
        # Apply brightness and contrast adjustments
        brightness = self.brightness_var.get()
        contrast = self.contrast_var.get()
        f = f * contrast + brightness
        f = np.clip(f, 0, 255)
        
        # Apply standard Gaussian blur if requested
        ksize = self.blur_var.get()
        if ksize > 0:
            if ksize % 2 == 0:
                ksize += 1
            f = cv2.GaussianBlur(f, (ksize, ksize), 0)
        
        # Apply low pass filter if strength > 0
        low_pass_strength = self.low_pass_var.get()
        if low_pass_strength > 0:
            f = self.apply_low_pass_filter(f, low_pass_strength)
        
        # Apply high pass filter if strength > 0
        high_pass_strength = self.high_pass_var.get()
        if high_pass_strength > 0:
            f = self.apply_high_pass_filter(f, high_pass_strength)
        
        # Apply Kalman filtering (temporal smoothing) if enabled
        if self.kalman_var.get():
            if self.prev_filtered_frame is None:
                self.prev_filtered_frame = f
            alpha_kalman = 0.7
            f = alpha_kalman * self.prev_filtered_frame + (1 - alpha_kalman) * f
            self.prev_filtered_frame = f
            
        return f.astype(np.uint8)
        
    def reset_kalman(self):
        self.prev_filtered_frame = None
        self.app.update_display()

class FileManager:
    """
    Handles file operations (open, save, drag-and-drop).
    """
    def __init__(self, app):
        self.app = app
        self.output_dir_prompted = False
        
    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Select TIF file",
            filetypes=[("TIF files", "*.tif;*.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.load_file(file_path)
            
    def load_file(self, file_path):
        try:
            global CURRENT_TIMESTAMPED_DIR, TIMESTAMPED_DIRNAME
            CURRENT_TIMESTAMPED_DIR = None
            TIMESTAMPED_DIRNAME = None
            progress = ProgressDialog(self.app.root, "Loading TIF File")
            progress.update_progress(0, 100, 0)
            progress.update()
            with Image.open(file_path) as img:
                frame_count = 0
                try:
                    while True:
                        img.seek(frame_count)
                        frame_count += 1
                        if frame_count % 10 == 0:
                            progress.update_progress(0, 100, frame_count)
                            progress.update()
                except EOFError:
                    pass
                img.seek(0)
                frames = []
                masks = []
                loaded_frames = 0
                try:
                    while True:
                        progress.update_progress(loaded_frames, frame_count, loaded_frames)
                        frame = np.array(img)
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        elif len(frame.shape) == 3 and frame.shape[2] == 1:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        frames.append(frame)
                        masks.append(np.zeros(frame.shape[:2], dtype=np.uint8))
                        loaded_frames += 1
                        img.seek(img.tell() + 1)
                        if loaded_frames % 10 == 0:
                            progress.update()
                except EOFError:
                    pass
            progress.update_progress(frame_count, frame_count, loaded_frames)
            progress.status_label.config(text="Finalizing...")
            progress.update()
            if not frames:
                progress.close()
                messagebox.showerror("Error", "No valid frames found in the TIF file.")
                return False
            self.app.set_file_data(file_path, frames, masks)
            if frames:
                self.app.resize_window_to_image(frames[0])
            progress.close()
            self.output_dir_prompted = False
            self.set_output_folder()
            if self.app.output_dir and not CURRENT_TIMESTAMPED_DIR:
                self._create_timestamped_directory()
            return True
        except Exception as e:
            try:
                progress.close()
            except:
                pass
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            return False
            
    def set_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder for Masks")
        if folder:
            self.app.output_dir = folder
            if self.app.current_file:
                self._create_timestamped_directory()
                if CURRENT_TIMESTAMPED_DIR:
                    messagebox.showinfo("Output Folder", f"Base output folder:\n{folder}\n\nData will be saved to:\n{CURRENT_TIMESTAMPED_DIR}")
                else:
                    messagebox.showinfo("Output Folder", f"Output will be saved to:\n{folder}")
            else:
                messagebox.showinfo("Output Folder", f"Output will be saved to:\n{folder}")
            self.output_dir_prompted = True
            
    def _create_timestamped_directory(self):
        global CURRENT_TIMESTAMPED_DIR, TIMESTAMPED_DIRNAME
        if not self.app.output_dir or not self.app.current_file:
            return
        if CURRENT_TIMESTAMPED_DIR is not None:
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(self.app.current_file))[0]
        TIMESTAMPED_DIRNAME = f"{base_filename}_{timestamp}"
        CURRENT_TIMESTAMPED_DIR = os.path.join(self.app.output_dir, TIMESTAMPED_DIRNAME)
        images_dir = os.path.join(CURRENT_TIMESTAMPED_DIR, "Images")
        masks_dir = os.path.join(CURRENT_TIMESTAMPED_DIR, "Masks")
        try:
            os.makedirs(CURRENT_TIMESTAMPED_DIR, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create output directories: {str(e)}")
            CURRENT_TIMESTAMPED_DIR = None
            TIMESTAMPED_DIRNAME = None
            
    def ensure_output_dirs_exist(self):
        global CURRENT_TIMESTAMPED_DIR
        if not self.app.output_dir:
            return False
        if not self.app.current_file:
            return False
        if CURRENT_TIMESTAMPED_DIR is None:
            self._create_timestamped_directory()
        if CURRENT_TIMESTAMPED_DIR is None or not os.path.exists(CURRENT_TIMESTAMPED_DIR):
            return False
        images_dir = os.path.join(CURRENT_TIMESTAMPED_DIR, "Images")
        masks_dir = os.path.join(CURRENT_TIMESTAMPED_DIR, "Masks")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        return True
            
    def save_all_masks(self):
        global CURRENT_TIMESTAMPED_DIR
        if not self.app.has_frames():
            messagebox.showwarning("Warning", "Please load a TIF file first")
            return
        if not self.app.output_dir:
            messagebox.showinfo("No Output Directory", "You haven't selected an output directory yet. Please select one now to save your masks.")
            self.set_output_folder()
            if not self.app.output_dir:
                messagebox.showwarning("Warning", "No output directory selected. Operation cancelled.")
                return
        if not self.ensure_output_dirs_exist():
            messagebox.showwarning("Warning", "Failed to create output directories")
            return
        images_dir = os.path.join(CURRENT_TIMESTAMPED_DIR, "Images")
        masks_dir = os.path.join(CURRENT_TIMESTAMPED_DIR, "Masks")
        filename = os.path.basename(self.app.current_file)
        base_filename = os.path.splitext(filename)[0]
        progress = ProgressDialog(self.app.root, "Saving Masks")
        total_frames = len(self.app.frames)
        try:
            saved_paths = []
            for i, (frame, mask) in enumerate(zip(self.app.frames, self.app.masks)):
                if i % 5 == 0:
                    progress.update_progress(i, total_frames, i)
                frame_path = os.path.join(images_dir, f"{base_filename}_frame_{i}.tif")
                pil_frame = Image.fromarray(frame)
                pil_frame.save(frame_path, compression="tiff_deflate")
                mask_path = os.path.join(masks_dir, f"{base_filename}_frame_{i}.tif")
                pil_mask = Image.fromarray(mask)
                pil_mask.save(mask_path, compression="tiff_deflate")
                saved_paths.append(mask_path)
            progress.update_progress(total_frames, total_frames, total_frames)
            progress.status_label.config(text="Creating image stack...")
            progress.update()
            images_stack_path = os.path.join(CURRENT_TIMESTAMPED_DIR, f"{base_filename}_images_stack.tif")
            masks_stack_path = os.path.join(CURRENT_TIMESTAMPED_DIR, f"{base_filename}_masks_stack.tif")
            pil_frames = [Image.fromarray(f) for f in self.app.frames]
            if pil_frames:
                pil_frames[0].save(images_stack_path, save_all=True, append_images=pil_frames[1:], compression="tiff_deflate")
            progress.status_label.config(text="Creating mask stack...")
            progress.update()
            pil_masks = [Image.fromarray(m) for m in self.app.masks]
            if pil_masks:
                pil_masks[0].save(masks_stack_path, save_all=True, append_images=pil_masks[1:], compression="tiff_deflate")
            progress.close()
            msg = f"Data saved to:\n{CURRENT_TIMESTAMPED_DIR}\n\nImages saved to:\n{images_dir}\n\nMasks saved to:\n{masks_dir}\n\nTotal frames: {len(saved_paths)}\n\nImage stack: {images_stack_path}\nMask stack: {masks_stack_path}"
            messagebox.showinfo("Success", msg)
        except Exception as e:
            try:
                progress.close()
            except:
                pass
            messagebox.showerror("Error", f"Failed to save masks: {str(e)}")
            
    def save_current_mask(self):
        global CURRENT_TIMESTAMPED_DIR
        if not self.app.current_file:
            return False
        if not self.app.output_dir and not self.output_dir_prompted:
            messagebox.showinfo("No Output Directory", "You haven't selected an output directory yet. Please select one now to save your masks.")
            self.set_output_folder()
            self.output_dir_prompted = True
            if not self.app.output_dir:
                messagebox.showwarning("Warning", "No output directory selected. Using default location.")
        current_frame = self.app.current_frame()
        current_mask = self.app.current_mask()
        current_idx = self.app.current_frame_idx
        filename = os.path.basename(self.app.current_file)
        base_filename = os.path.splitext(filename)[0]
        if self.app.output_dir:
            if not self.ensure_output_dirs_exist():
                messagebox.showwarning("Warning", "Output directory structure not found. Attempting to recreate it.")
                if CURRENT_TIMESTAMPED_DIR is None:
                    self._create_timestamped_directory()
                    if CURRENT_TIMESTAMPED_DIR is None:
                        base_path = os.path.splitext(self.app.current_file)[0]
                        mask_path = f"{base_path}_mask_{current_idx}.tif"
                        pil_mask = Image.fromarray(current_mask)
                        pil_mask.save(mask_path, compression="tiff_deflate")
                        messagebox.showwarning("Warning", f"Saving to default location: {mask_path}")
                        return True
                images_dir = os.path.join(CURRENT_TIMESTAMPED_DIR, "Images")
                masks_dir = os.path.join(CURRENT_TIMESTAMPED_DIR, "Masks")
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(masks_dir, exist_ok=True)
            else:
                images_dir = os.path.join(CURRENT_TIMESTAMPED_DIR, "Images")
                masks_dir = os.path.join(CURRENT_TIMESTAMPED_DIR, "Masks")
            img_path = os.path.join(images_dir, f"{base_filename}_frame_{current_idx}.tif")
            mask_path = os.path.join(masks_dir, f"{base_filename}_frame_{current_idx}.tif")
            try:
                pil_frame = Image.fromarray(current_frame)
                pil_frame.save(img_path, compression="tiff_deflate")
                pil_mask = Image.fromarray(current_mask)
                pil_mask.save(mask_path, compression="tiff_deflate")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save to specified location: {str(e)}")
                base_path = os.path.splitext(self.app.current_file)[0]
                mask_path = f"{base_path}_mask_{current_idx}.tif"
                pil_mask = Image.fromarray(current_mask)
                pil_mask.save(mask_path, compression="tiff_deflate")
                messagebox.showinfo("Info", f"Saved mask to: {mask_path}")
            return True
        else:
            base_path = os.path.splitext(self.app.current_file)[0]
            mask_path = f"{base_path}_mask_{current_idx}.tif"
            pil_mask = Image.fromarray(current_mask)
            pil_mask.save(mask_path, compression="tiff_deflate")
            return True


class DragDropHandler:
    """
    Handles drag and drop functionality for the application.
    """
    def __init__(self, root, app):
        self.root = root
        self.app = app
        if sys.platform == 'win32':
            self.root.drop_target_register(1)
            self.root.dnd_bind('<<Drop>>', self._handle_drop)
        else:
            self.root.bind("<Drop>", self._handle_drop)
            
    def _handle_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            if file_path.lower().endswith(('.tif', '.tiff')):
                if self.app.file_manager.load_file(file_path):
                    self.app.file_manager.set_output_folder()
            else:
                messagebox.showwarning("Invalid File", "Please drop a TIF file (*.tif, *.tiff)")


# Part 3/5: Intelligent Frame Selection & Frame Selector Dialog

class IntelligentFrameSelector:
    """
    Intelligent Frame Selection for Manual Labeling.
    Uses optical flow and scene change detection to select key frames.
    """
    def __init__(self, app, flow_method='dual_tv_l1', error_threshold=0.15, min_frames=2, max_frames=None):
        self.app = app
        self.flow_method = flow_method
        self.error_threshold = error_threshold
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.selected_frames = []
        self.frame_scores = {}
        self.scene_changes = []
        
    def compute_optical_flow(self, frame1, frame2):
        try:
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
            flow = optical_flow.calc(frame1, frame2, None)
        except Exception as e:
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
        return flow
    
    def calculate_flow_statistics(self, flow):
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        du_dx = cv2.Sobel(flow[..., 0], cv2.CV_64F, 1, 0, ksize=3)
        dv_dy = cv2.Sobel(flow[..., 1], cv2.CV_64F, 0, 1, ksize=3)
        flow_div = du_dx + dv_dy
        dv_dx = cv2.Sobel(flow[..., 1], cv2.CV_64F, 1, 0, ksize=3)
        du_dy = cv2.Sobel(flow[..., 0], cv2.CV_64F, 0, 1, ksize=3)
        flow_curl = dv_dx - du_dy
        stats = {
            'mean_mag': np.mean(flow_mag),
            'max_mag': np.max(flow_mag),
            'p95_mag': np.percentile(flow_mag, 95),
            'mean_div_abs': np.mean(np.abs(flow_div)),
            'mean_curl_abs': np.mean(np.abs(flow_curl)),
            'large_motion_ratio': np.mean(flow_mag > 10.0),
        }
        return stats
    
    def estimate_propagation_error(self, frame_idx, video_frames, labeled_frames=None):
        if labeled_frames is None or len(labeled_frames) == 0:
            n_frames = len(video_frames)
            return abs(frame_idx - n_frames // 2) / n_frames
        nearest_labeled = min(labeled_frames, key=lambda x: abs(x - frame_idx))
        distance = abs(nearest_labeled - frame_idx)
        max_flow_distance = 5
        if distance <= max_flow_distance:
            frame1 = cv2.cvtColor(video_frames[nearest_labeled], cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(video_frames[frame_idx], cv2.COLOR_RGB2GRAY)
            flow = self.compute_optical_flow(frame1, frame2)
            flow_stats = self.calculate_flow_statistics(flow)
        else:
            flow_stats_cumulative = {
                'mean_mag': 0,
                'max_mag': 0,
                'p95_mag': 0,
                'mean_div_abs': 0,
                'mean_curl_abs': 0,
                'large_motion_ratio': 0
            }
            step = 1 if frame_idx > nearest_labeled else -1
            for i in range(nearest_labeled, frame_idx, step):
                idx1 = i
                idx2 = i + step
                frame1 = cv2.cvtColor(video_frames[idx1], cv2.COLOR_RGB2GRAY)
                frame2 = cv2.cvtColor(video_frames[idx2], cv2.COLOR_RGB2GRAY)
                flow = self.compute_optical_flow(frame1, frame2)
                flow_stats = self.calculate_flow_statistics(flow)
                for key in flow_stats_cumulative:
                    flow_stats_cumulative[key] += flow_stats[key]
            flow_stats = flow_stats_cumulative
            for key in flow_stats:
                flow_stats[key] /= distance
        error_score = (
            0.4 * flow_stats['mean_mag'] +
            0.2 * flow_stats['p95_mag'] / 50.0 +
            0.2 * flow_stats['mean_div_abs'] +
            0.1 * flow_stats['mean_curl_abs'] +
            0.1 * flow_stats['large_motion_ratio'] * 10.0
        )
        error_score *= np.sqrt(distance)
        return error_score
    
    def detect_scene_changes(self, video_frames, threshold=40.0):
        n_frames = len(video_frames)
        scene_changes = []
        diffs = []
        for i in range(1, n_frames):
            prev_frame = cv2.cvtColor(video_frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_frame = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(prev_frame, curr_frame)
            diff_mean = np.mean(diff)
            diffs.append(diff_mean)
        if len(diffs) > 5:
            try:
                diffs_smooth = savgol_filter(diffs, 5, 2)
            except:
                diffs_smooth = diffs
        else:
            diffs_smooth = diffs
        for i in range(1, len(diffs_smooth) - 1):
            if diffs_smooth[i] > diffs_smooth[i-1] and diffs_smooth[i] > diffs_smooth[i+1]:
                if diffs_smooth[i] > threshold:
                    scene_changes.append(i + 1)
        return scene_changes
    
    def select_frames(self, progress_callback=None):
        import time
        if not self.app.frames:
            return []
        video_frames = self.app.frames
        n_frames = len(video_frames)
        start_time = time.time()
        if progress_callback:
            progress_callback(0, f"Starting analysis of {n_frames} frames...")
        if progress_callback:
            progress_callback(10, "Detecting scene changes...")
        self.scene_changes = self.detect_scene_changes(video_frames)
        if progress_callback:
            progress_callback(20, f"Found {len(self.scene_changes)} scene changes")
        if progress_callback:
            progress_callback(25, "Computing frame differences...")
        if self.max_frames is None:
            budget = max(self.min_frames, int(np.sqrt(n_frames) * 0.5))
        else:
            budget = min(self.max_frames, n_frames)
        budget = min(budget, 50)
        if progress_callback:
            progress_callback(30, f"Planning to select {budget} frames...")
        selected_frames = [0]  # Always label the first frame
        labeled_frames = [0]
        error_estimates = [0]
        for j in range(1, n_frames):
            err = self.estimate_propagation_error(j, video_frames, labeled_frames)
            error_estimates.append(err)
        mean_err = np.mean(error_estimates)
        std_err = np.std(error_estimates)
        adaptive_threshold = mean_err + 0.5 * std_err
        if progress_callback:
            progress_callback(35, f"Adaptive threshold set to {adaptive_threshold:.2f}")
        for j in range(1, n_frames):
            if error_estimates[j] > adaptive_threshold:
                selected_frames.append(j)
                labeled_frames.append(j)
                for k in range(j+1, n_frames):
                    error_estimates[k] = self.estimate_propagation_error(k, video_frames, labeled_frames)
        if len(selected_frames) > budget:
            selected_frames = selected_frames[::max(1, len(selected_frames) // budget)]
        selected_frames = sorted(list(set(selected_frames)))
        self.selected_frames = selected_frames
        self.frame_scores = {frame: error_estimates[frame] for frame in selected_frames}
        if progress_callback:
            total_time = time.time() - start_time
            progress_callback(100, f"Selection complete: {len(selected_frames)} frames in {total_time:.1f}s")
        return selected_frames


class FrameSelectorDialog:
    """Dialog for intelligent frame selection and key frame review."""
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Intelligent Frame Selection")
        self.dialog.configure(bg='#333333')
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.dialog.geometry("500x400")
        self.center_dialog()
        self.create_widgets()
        self.selected_frames = []
        self.selection_in_progress = False
        self.original_frame_idx = self.app.current_frame_idx
        
    def center_dialog(self):
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = self.parent.winfo_rootx() + (self.parent.winfo_width() - width) // 2
        y = self.parent.winfo_rooty() + (self.parent.winfo_height() - height) // 2
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
    def create_widgets(self):
        self.main_frame = ttk.Frame(self.dialog, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.main_frame, text="Intelligent Frame Selection", font=("Arial", 16, "bold")).pack(pady=(0, 10))
        description = ("This tool automatically selects the most informative frames for manual labeling. "
                       "It analyzes video content to identify frames that are most different from each other.")
        ttk.Label(self.main_frame, text=description, wraplength=450, justify="center").pack(pady=(0, 20))
        settings_frame = ttk.LabelFrame(self.main_frame, text="Selection Settings")
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        max_frames_frame = ttk.Frame(settings_frame)
        max_frames_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(max_frames_frame, text="Max frames to select:").pack(side=tk.LEFT)
        self.max_frames_var = tk.IntVar(value=10)
        self.max_frames_spin = ttk.Spinbox(max_frames_frame, from_=2, to=100, width=5, textvariable=self.max_frames_var)
        self.max_frames_spin.pack(side=tk.LEFT, padx=(5, 0))
        self.auto_frames_var = tk.BooleanVar(value=True)
        auto_frames_check = ttk.Checkbutton(max_frames_frame, text="Auto (adaptive)", variable=self.auto_frames_var, command=self.toggle_max_frames)
        auto_frames_check.pack(side=tk.LEFT, padx=(15, 0))
        self.max_frames_spin.configure(state="disabled")
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(threshold_frame, text="Error threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=0.15)
        ttk.Scale(threshold_frame, from_=0.05, to=0.25, variable=self.threshold_var, length=200).pack(side=tk.LEFT, padx=(5,5), fill=tk.X, expand=True)
        ttk.Label(threshold_frame, textvariable=tk.StringVar(value=lambda: f"{self.threshold_var.get():.2f}")).pack(side=tk.LEFT, padx=(0,5))
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_label = ttk.Label(self.progress_frame, text="Analyzing video...", font=("Arial", 10))
        self.progress_label.pack(pady=(10, 5))
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=5, fill=tk.X)
        self.status_label = ttk.Label(self.progress_frame, text="", font=("Arial", 9))
        self.status_label.pack(pady=(0, 10))
        self.results_frame = ttk.Frame(self.main_frame)
        ttk.Label(self.results_frame, text="Selected Frames:", font=("Arial", 12, "bold")).pack(pady=(10, 5))
        scroll_frame = ttk.Frame(self.results_frame)
        scroll_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        scrollbar = ttk.Scrollbar(scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text = tk.Text(scroll_frame, width=40, height=10, yscrollcommand=scrollbar.set, bg="#444444", fg="white", font=("Courier", 10))
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)
        # Key Frame List Panel on the right
        self.key_frame_panel = ttk.Frame(self.main_frame)
        self.key_frame_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        ttk.Label(self.key_frame_panel, text="Key Frames", font=("Arial", 12, "bold")).pack(pady=(0,5))
        self.key_frame_list = tk.Listbox(self.key_frame_panel, width=12, height=20, bg="#444444", fg="white", font=("Courier", 10), selectbackground="#2a6099")
        self.key_frame_list.pack(fill=tk.Y, expand=True)
        self.key_frame_list.bind("<<ListboxSelect>>", self.on_key_frame_select)
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        self.start_button = ttk.Button(button_frame, text="Start Selection", command=self.start_selection, width=20)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.on_cancel, width=20)
        self.cancel_button.pack(side=tk.RIGHT)
        self.nav_button_frame = ttk.Frame(self.main_frame)
        self.prev_button = ttk.Button(self.nav_button_frame, text="← Previous", command=self.goto_prev_selected, width=15)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = ttk.Button(self.nav_button_frame, text="Next →", command=self.goto_next_selected, width=15)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.jump_button = ttk.Button(self.nav_button_frame, text="Jump to Selected", command=self.show_jump_dialog, width=15)
        self.jump_button.pack(side=tk.LEFT, padx=5)
        self.close_button = ttk.Button(self.nav_button_frame, text="Close", command=self.dialog.destroy, width=15)
        self.close_button.pack(side=tk.RIGHT, padx=5)
        self.current_selected_idx = 0
        
    def toggle_max_frames(self):
        if self.auto_frames_var.get():
            self.max_frames_var.set(0)
            self.max_frames_spin.configure(state="disabled")
        else:
            if self.max_frames_var.get() == 0:
                self.max_frames_var.set(10)
            self.max_frames_spin.configure(state="normal")
            
    def start_selection(self):
        if not self.app.has_frames():
            messagebox.showerror("Error", "No frames loaded. Please load a TIF file first.")
            return
        self.selection_in_progress = True
        self.selection_start_time = time.time()
        self.main_frame.winfo_children()[2].pack_forget()  # Hide settings frame
        self.main_frame.winfo_children()[3].pack_forget()  # Hide button frame
        self.progress_frame.pack(fill=tk.X, padx=10, pady=10)
        self.progress_bar["value"] = 0
        self.progress_label.config(text="Initializing...")
        self.status_label.config(text="")
        self.dialog.update_idletasks()
        max_frames = None if self.auto_frames_var.get() else self.max_frames_var.get()
        threshold = self.threshold_var.get()
        self.frame_selector = IntelligentFrameSelector(app=self.app, error_threshold=threshold, min_frames=2, max_frames=max_frames)
        self.selection_thread = threading.Thread(target=self.run_selection, args=(self.frame_selector,))
        self.selection_thread.daemon = True
        self.selection_thread.start()
        self.dialog.after(100, self.check_selection_status)
        
    def run_selection(self, selector):
        try:
            self.selected_frames = selector.select_frames(self.update_progress)
            time.sleep(0.5)
        except Exception as e:
            self.selection_error = str(e)
        finally:
            self.selection_in_progress = False
            
    def update_progress(self, progress, status_text):
        self.dialog.after(0, lambda: self._update_progress_ui(progress, status_text))
        
    def _update_progress_ui(self, progress, status_text):
        self.progress_bar["value"] = progress
        self.status_label.config(text=status_text)
        self.progress_label.config(text=f"Analyzing video... {int(progress)}%")
        if not hasattr(self, 'progress_log'):
            log_frame = ttk.Frame(self.progress_frame)
            log_frame.pack(pady=5, fill=tk.X)
            scrollbar = ttk.Scrollbar(log_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.progress_log = tk.Text(log_frame, width=50, height=5, bg="#333333", fg="#90EE90", font=("Courier", 9))
            self.progress_log.pack(side=tk.LEFT, fill=tk.X, expand=True)
            scrollbar.config(command=self.progress_log.yview)
            self.progress_log.config(yscrollcommand=scrollbar.set)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.progress_log.config(state=tk.NORMAL)
        self.progress_log.insert(tk.END, f"[{timestamp}] {status_text}\n")
        self.progress_log.see(tk.END)
        self.progress_log.config(state=tk.DISABLED)
        self.dialog.update_idletasks()
        
    def check_selection_status(self):
        if self.selection_in_progress:
            self.dialog.after(100, self.check_selection_status)
        else:
            self.on_selection_complete()
            
    def on_selection_complete(self):
        self.progress_frame.pack_forget()
        self.show_selection_results()
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.nav_button_frame.pack(fill=tk.X, pady=(10, 0))
        if self.selected_frames:
            self.current_selected_idx = 0
            self.goto_selected_frame(self.selected_frames[0])
            
    def show_selection_results(self):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Selected {len(self.selected_frames)} frames:\n\n")
        for i, frame_idx in enumerate(self.selected_frames):
            score_text = f" (Score: {self.frame_selector.frame_scores.get(frame_idx, 0):.2f})"
            scene_change_text = " [Scene Change]" if frame_idx in self.frame_selector.scene_changes else ""
            self.results_text.insert(tk.END, f"{i+1}. Frame {frame_idx+1}{scene_change_text}{score_text}\n")
            if i == self.current_selected_idx:
                self.results_text.tag_add("current", f"{i+3}.0", f"{i+3}.end")
                self.results_text.tag_config("current", background="#2a6099", foreground="white")
        self.results_text.config(state=tk.DISABLED)
        # Update key frame list panel
        self.key_frame_list.delete(0, tk.END)
        for i, frame_idx in enumerate(self.selected_frames):
            label_text = f"Frame {frame_idx+1}"
            if frame_idx in LABELED_KEYFRAMES:
                label_text += " ✓"
            self.key_frame_list.insert(tk.END, label_text)
            
    def goto_selected_frame(self, frame_idx):
        if 0 <= frame_idx < len(self.app.frames):
            self.app.set_current_frame(frame_idx)
            self.results_text.config(state=tk.NORMAL)
            self.results_text.tag_remove("current", "1.0", tk.END)
            try:
                selected_idx = self.selected_frames.index(frame_idx)
                self.current_selected_idx = selected_idx
                line_num = selected_idx + 3
                self.results_text.tag_add("current", f"{line_num}.0", f"{line_num}.end")
                self.results_text.tag_config("current", background="#2a6099", foreground="white")
                self.results_text.see(f"{line_num}.0")
            except ValueError:
                pass
            self.results_text.config(state=tk.DISABLED)
            
    def goto_next_selected(self):
        if not self.selected_frames:
            return
        self.current_selected_idx = (self.current_selected_idx + 1) % len(self.selected_frames)
        self.goto_selected_frame(self.selected_frames[self.current_selected_idx])
        
    def goto_prev_selected(self):
        if not self.selected_frames:
            return
        self.current_selected_idx = (self.current_selected_idx - 1) % len(self.selected_frames)
        self.goto_selected_frame(self.selected_frames[self.current_selected_idx])
        
    def show_jump_dialog(self):
        if not self.selected_frames:
            return
        jump_dialog = tk.Toplevel(self.dialog)
        jump_dialog.title("Jump to Selected Frame")
        jump_dialog.configure(bg='#333333')
        jump_dialog.transient(self.dialog)
        jump_dialog.grab_set()
        frame = ttk.Frame(jump_dialog, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Select a frame to jump to:", font=("Arial", 11)).pack(pady=(0, 10))
        listbox = tk.Listbox(frame, width=40, height=15, bg="#444444", fg="white", font=("Courier", 10), selectbackground="#2a6099")
        listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        for i, frame_idx in enumerate(self.selected_frames):
            scene_mark = "🎬" if frame_idx in self.frame_selector.scene_changes else "  "
            listbox.insert(tk.END, f"{scene_mark} Frame {frame_idx+1}")
            if i == self.current_selected_idx:
                listbox.selection_set(i)
                listbox.see(i)
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_frame, text="Jump", command=lambda: self._jump_to_selected(listbox.curselection(), jump_dialog), width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=jump_dialog.destroy, width=15).pack(side=tk.RIGHT, padx=5)
        jump_dialog.update_idletasks()
        width = jump_dialog.winfo_width()
        height = jump_dialog.winfo_height()
        x = self.dialog.winfo_rootx() + (self.dialog.winfo_width() - width) // 2
        y = self.dialog.winfo_rooty() + (self.dialog.winfo_height() - height) // 2
        jump_dialog.geometry(f"{width}x{height}+{x}+{y}")
        
    def _jump_to_selected(self, selection, dialog):
        if selection:
            idx = selection[0]
            if 0 <= idx < len(self.selected_frames):
                self.current_selected_idx = idx
                self.goto_selected_frame(self.selected_frames[idx])
                dialog.destroy()
        
    def on_key_frame_select(self, event):
        selection = self.key_frame_list.curselection()
        if selection:
            idx = selection[0]
            frame_idx = self.selected_frames[idx]
            self.goto_selected_frame(frame_idx)
            
    def on_cancel(self):
        if self.selection_in_progress:
            if messagebox.askyesno("Cancel Selection", "Frame selection is in progress. Are you sure you want to cancel?"):
                self.dialog.destroy()
        else:
            self.app.set_current_frame(self.original_frame_idx)
            self.dialog.destroy()
            
    def export_selected_frames(self):
        if not self.selected_frames:
            messagebox.showinfo("Info", "No frames have been selected yet.")
            return
        file_path = filedialog.asksaveasfilename(parent=self.dialog, title="Save Selected Frames",
                                                 filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                                                 defaultextension=".txt")
        if not file_path:
            return
        try:
            with open(file_path, 'w') as f:
                f.write("# Selected frames for manual labeling\n")
                f.write(f"# File: {self.app.current_file}\n")
                f.write(f"# Total frames selected: {len(self.selected_frames)}\n\n")
                for i, frame_idx in enumerate(self.selected_frames):
                    scene_change = " [Scene Change]" if frame_idx in self.frame_selector.scene_changes else ""
                    f.write(f"Frame {frame_idx+1}{scene_change}\n")
            messagebox.showinfo("Success", f"Selected frames saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save frames: {str(e)}")

# Part 4/5: Main Application Class (TIFLabeler) with Propagation, Undo, and Loop Playback
# Part 4/5: Main Application Class (TIFLabeler) with Propagation, Undo, and Loop Playback

class TIFLabeler:

    def __init__(self, root):
        self.root = root
        self.setup_theme()
        self.current_file = None
        self.frames = []
        self.masks = []
        self.current_frame_idx = 0
        self.tool = "draw"
        self.brush_size = 5
        self.output_dir = None
        self.display_info = None
        self.previous_mask = None  # For undo feature
        self.labeled_keyframes = set()
        self.button_width = 35  # Default button width for right panel
        
        # Main application frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top toolbar frame
        self.toolbar_frame = ttk.Frame(self.main_frame)
        self.toolbar_frame.pack(fill=tk.X, pady=5)
        
        # Modified toolbar - we'll keep some controls but move buttons to right panel
        self.toolbar = ttk.Frame(self.toolbar_frame)
        self.toolbar.pack(fill=tk.X, expand=True)
        
        # Tool indicator in top toolbar
        self.tool_indicator = ttk.Label(self.toolbar, text="Active: Draw", width=15)
        self.tool_indicator.pack(side=tk.LEFT, padx=10)
        
        # Add brush size control to top-right
        self.brush_size_frame = ttk.Frame(self.toolbar)
        self.brush_size_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Label(self.brush_size_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        self.brush_scale = ttk.Scale(self.brush_size_frame, from_=1, to=20, orient=tk.HORIZONTAL, 
                                   command=self.update_brush_size, length=150)
        self.brush_scale.set(5)
        self.brush_scale.pack(side=tk.LEFT, padx=5)
        
        # Undo button in top toolbar
        self.undo_btn = ttk.Button(self.toolbar, text="Undo", command=self.undo_last_action, width=10)
        self.undo_btn.pack(side=tk.LEFT, padx=5)
        
        # Zoom controls on right side of top toolbar
        zoom_frame = ttk.Frame(self.toolbar)
        zoom_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Button(zoom_frame, text="Zoom +", width=8, command=lambda: self.zoom_in()).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Zoom -", width=8, command=lambda: self.zoom_out()).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset View", width=10, command=lambda: self.reset_view()).pack(side=tk.LEFT, padx=2)
        
        # Create a horizontal paned window for split layout (canvas + right panel)
        self.split_pane = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.split_pane.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Canvas frame
        self.canvas_frame = ttk.Frame(self.split_pane)
        self.split_pane.add(self.canvas_frame, weight=4)  # Canvas gets more space
        
        # Right side - Button panel
        self.right_panel = ttk.Frame(self.split_pane)
        self.split_pane.add(self.right_panel, weight=1)  # Button panel gets less space
        
        # Set up the canvas
        self.img_canvas = ImageCanvas(self.canvas_frame, self)
        self.img_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Set up the right panel with buttons in specified order
        self.right_panel_buttons = ttk.Frame(self.right_panel, padding=10)
        self.right_panel_buttons.pack(fill=tk.BOTH, expand=True)
        
        # Add title for right panel
        ttk.Label(self.right_panel_buttons, text="Tools", font=("Arial", 12, "bold")).pack(pady=(0, 15))

        # Add button size controls
        self.button_size_frame = ttk.Frame(self.right_panel_buttons)
        self.button_size_frame.pack(pady=(0, 15), fill=tk.X)
        ttk.Label(self.button_size_frame, text="Button Size:").pack(side=tk.LEFT)

        # Decrease button
        self.decrease_size_btn = tk.Button(self.button_size_frame, text="-", width=3,
                                        command=self.decrease_button_size, bg="#555555", fg="white")
        self.decrease_size_btn.pack(side=tk.LEFT, padx=2)

        # Increase button
        self.increase_size_btn = tk.Button(self.button_size_frame, text="+", width=3,
                                        command=self.increase_button_size, bg="#555555", fg="white")
        self.increase_size_btn.pack(side=tk.LEFT, padx=2)


        # # 1. Draw button
        self.draw_btn = tk.Button(self.right_panel_buttons, text="Draw", width=self.button_width,
            height=5, 
                               command=lambda: self.set_tool("draw"), bg="#555555", fg="white")
        self.draw_btn.pack(pady=5, fill=tk.X)

        # 2. Erase button
        self.erase_btn = tk.Button(self.right_panel_buttons, text="Erase", width=self.button_width,
                               height=5, 
                                command=lambda: self.set_tool("erase"), bg="#555555", fg="white")
        self.erase_btn.pack(pady=5, fill=tk.X)

        # 3. Clear Mask button
        self.clear_btn = tk.Button(self.right_panel_buttons, text="Clear Mask", width=self.button_width,
                                height=5, 
                                command=self.clear_mask, bg="#555555", fg="white")
        self.clear_btn.pack(pady=5, fill=tk.X)

        # 4. Loop control
        self.loop_span_var = tk.StringVar(value="30")
        self.loop_frame = ttk.Frame(self.right_panel_buttons)
        self.loop_frame.pack(pady=5, fill=tk.X)

        ttk.Label(self.loop_frame, text="Loop Span:").pack(side=tk.LEFT)
        self.loop_span_spin = ttk.Spinbox(self.loop_frame, from_=5, to=200, width=5, 
                                       textvariable=self.loop_span_var)
        self.loop_span_spin.pack(side=tk.LEFT, padx=2)

        self.loop_btn = tk.Button(self.right_panel_buttons, text="Loop", width=self.button_width,
                              height=5, 
                               command=self.loop_playback, bg="#555555", fg="white")
        self.loop_btn.pack(pady=5, fill=tk.X)

        # 5. Generate Mask button
        self.generate_btn = tk.Button(self.right_panel_buttons, text="Generate Mask", width=self.button_width,
                                  height=5, 
                                   command=self.generate_mask, bg="#555555", fg="white")
        self.generate_btn.pack(pady=5, fill=tk.X)

        # # And for the size adjustment buttons:
        # self.decrease_size_btn = tk.Button(self.button_size_frame, text="-", width=3,
        #                                 command=self.decrease_button_size, bg="#555555", fg="white")
        # self.decrease_size_btn.pack(side=tk.LEFT, padx=2)

        # self.increase_size_btn = tk.Button(self.button_size_frame, text="+", width=3,
        #                                 command=self.increase_button_size, bg="#555555", fg="white")
        # self.increase_size_btn.pack(side=tk.LEFT, padx=2)



        # Add navigation at the bottom
        self.nav_frame = ttk.Frame(self.main_frame)
        self.nav_frame.pack(fill=tk.X, pady=5)
        self.navigator = FrameNavigator(self.nav_frame, self)
        self.navigator.pack(fill=tk.X, expand=True)
        
        # Move the enhancement frame to bottom
        self.enhancement = ImageEnhancement(self.main_frame, self)
        self.enhancement.pack(fill=tk.X, padx=10, pady=10)
        
        # File manager and drag-drop handler
        self.file_manager = FileManager(self)
        try:
            self.drag_drop = DragDropHandler(self.root, self)
        except:
            pass
        
        # Setup menus and bindings
        self.setup_menu()
        self.setup_bindings()
        self.root.bind("<Configure>", self._on_window_resize)
        
    def _on_window_resize(self, event):
        if event.widget == self.root:
            self.update_display()
    

    def setup_theme(self):
        self.root.configure(bg='#333333')
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
            style.configure("TFrame", background='#333333')
            style.configure("TButton", background='#555555', foreground='white')
            style.configure("TLabel", background='#333333', foreground='white')
            style.configure("TLabelframe", background='#333333', foreground='white')
            style.configure("TLabelframe.Label", background='#333333', foreground='white')
            style.configure("TScale", background='#333333', troughcolor='#555555')
            style.configure("TCheckbutton", background='#333333', foreground='white')
            self.root.option_add("*Menu.Background", '#444444')
            self.root.option_add("*Menu.Foreground", 'white')
            self.root.option_add("*Menu.activeBackground", '#666666')
            self.root.option_add("*Menu.activeForeground", 'white')
        except:
            pass
            
    def setup_menu(self):
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)
        self.file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open", command=self.file_manager.open_file)
        self.file_menu.add_command(label="Set Output Folder", command=self.file_manager.set_output_folder)
        self.file_menu.add_command(label="Generate Mask (Ctrl+S)", command=self.generate_mask)
        self.file_menu.add_command(label="Save All Masks (Ctrl+A)", command=self.file_manager.save_all_masks)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)
        self.view_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_command(label="Zoom In", command=self.zoom_in)
        self.view_menu.add_command(label="Zoom Out", command=self.zoom_out)
        self.view_menu.add_command(label="Reset View", command=self.reset_view)
        self.view_menu.add_command(label="Center Image", command=self.img_canvas.center_image)
        self.view_menu.add_separator()
        self.view_menu.add_checkbutton(label="Debug Mode", command=lambda: self.img_canvas._toggle_debug(None))
        self.help_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        self.help_menu.add_command(label="About", command=self._show_about)
        
    def setup_bindings(self):
        self.root.bind("<Control-z>", lambda e: self.undo_last_action())
        self.root.bind("<Control-s>", lambda e: self.generate_mask())
        self.root.bind("<Control-a>", lambda e: self.file_manager.save_all_masks())
        self.root.bind("<Control-i>", lambda e: self.frame_selector_integration.show_frame_selector())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<plus>", lambda e: self.zoom_in())
        self.root.bind("<minus>", lambda e: self.zoom_out())
        self.root.bind("<Control-0>", lambda e: self.reset_view())
        self.root.bind("c", lambda e: self.img_canvas.center_image())
        
    def has_frames(self):
        return len(self.frames) > 0
        
    def current_frame(self):
        if not self.has_frames():
            return None
        return self.frames[self.current_frame_idx]
        
    def current_mask(self):
        if not self.has_frames():
            return None
        return self.masks[self.current_frame_idx]
        
    def set_file_data(self, file_path, frames, masks):
        self.current_file = file_path
        self.frames = frames
        self.masks = masks
        self.current_frame_idx = 0
        self.enhancement.reset_kalman()
        self.update_frame_info()
        self.img_canvas.zoom_factor = 1.0
        self.img_canvas.pan_x = 0
        self.img_canvas.pan_y = 0
        self.root.update_idletasks()
        self.update_display()
        self.root.after(200, self.update_display)

    def increase_button_size(self):
        self.button_width += 5
        self.update_button_sizes()
        
    def decrease_button_size(self):
        if self.button_width > 20:  # Prevent buttons from becoming too small
            self.button_width -= 5
            self.update_button_sizes()
            
    def update_button_sizes(self):
        # Update all button widths in the right panel
        self.draw_btn.config(width=self.button_width)
        self.erase_btn.config(width=self.button_width)
        self.clear_btn.config(width=self.button_width)
        self.loop_btn.config(width=self.button_width)
        self.generate_btn.config(width=self.button_width)
        
    def update_frame_info(self):
        total = len(self.frames) if self.frames else 0
        self.navigator.update_frame_label(self.current_frame_idx, total)
        self.navigator.update_slider_position(self.current_frame_idx, total)
        if self.current_file:
            filename = os.path.basename(self.current_file)
            self.root.title(f"TIF Manual Labeler v2.0 - {filename} - Frame {self.current_frame_idx+1}/{len(self.frames)}")
        
    def update_display(self, zoom_center_x=None, zoom_center_y=None, old_zoom=None):
        if not self.has_frames():
            return
        original_frame = self.current_frame().copy()
        filtered_frame = self.enhancement.apply_filters(original_frame)
        mask = self.current_mask()
        overlay = filtered_frame.copy()
        overlay[mask > 0] = [255, 0, 0]
        cv2.addWeighted(overlay, 0.3, filtered_frame, 0.7, 0, filtered_frame)
        image = Image.fromarray(filtered_frame)
        canvas_width = self.img_canvas.canvas.winfo_width()
        canvas_height = self.img_canvas.canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            self.root.update_idletasks()
            self.root.after(250, self.update_display)
            return
        zoom = self.img_canvas.zoom_factor
        img_height, img_width = filtered_frame.shape[:2]
        new_width = int(img_width * zoom)
        new_height = int(img_height * zoom)
        if new_width > 0 and new_height > 0:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.img_canvas.canvas.delete("all")
        x_pos = max(0, (canvas_width - new_width) // 2) + self.img_canvas.pan_x
        y_pos = max(0, (canvas_height - new_height) // 2) + self.img_canvas.pan_y
        self.display_info = (x_pos, y_pos, new_width, new_height)
        self.img_canvas.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=photo)
        self.img_canvas.canvas.image = photo
        if self.img_canvas.debug_mode:
            self.img_canvas.canvas.create_text(10, 10, text=f"Zoom: {zoom:.2f}x | Pan: ({self.img_canvas.pan_x}, {self.img_canvas.pan_y})", anchor="nw", fill="yellow")
            
    def zoom_in(self):
        if not self.has_frames():
            return
        old_zoom = self.img_canvas.zoom_factor
        self.img_canvas.zoom_factor = min(self.img_canvas.zoom_factor * 1.2, 10.0)
        canvas_width = self.img_canvas.canvas.winfo_width()
        canvas_height = self.img_canvas.canvas.winfo_height()
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        original_h, original_w = self.current_frame().shape[:2]
        old_img_x = original_w // 2
        old_img_y = original_h // 2
        self.img_canvas._update_zoom_pan(center_x, center_y, old_img_x, old_img_y)
        
    def zoom_out(self):
        if not self.has_frames():
            return
        old_zoom = self.img_canvas.zoom_factor
        self.img_canvas.zoom_factor = max(self.img_canvas.zoom_factor / 1.2, 0.1)
        canvas_width = self.img_canvas.canvas.winfo_width()
        canvas_height = self.img_canvas.canvas.winfo_height()
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        original_h, original_w = self.current_frame().shape[:2]
        old_img_x = original_w // 2
        old_img_y = original_h // 2
        self.img_canvas._update_zoom_pan(center_x, center_y, old_img_x, old_img_y)
        
    def reset_view(self):
        self.img_canvas.reset_view()
        
    def set_tool(self, tool):
        self.tool = tool
        self.tool_indicator.config(text=f"Active: {tool.capitalize()}")
            
    def update_brush_size(self, value):
        self.brush_size = int(float(value))
        
    def clear_mask(self):
        if not self.has_frames():
            return
        self.masks[self.current_frame_idx] = np.zeros_like(self.masks[self.current_frame_idx])
        self.update_display()
     



    def prev_frame(self):
        if not self.has_frames() or self.current_frame_idx <= 0:
            return
        self.current_frame_idx -= 1
        self.enhancement.reset_kalman()
        self.update_frame_info()
        self.update_display()
        
    def next_frame(self, skip=1):
        if not self.has_frames():
            return
        next_idx = self.current_frame_idx + skip
        if next_idx >= len(self.frames):
            next_idx = len(self.frames) - 1
        if next_idx > self.current_frame_idx:
            self.current_frame_idx = next_idx
            self.enhancement.reset_kalman()
            self.update_frame_info()
            self.update_display()
        
    def set_current_frame(self, frame_idx):
        if not self.has_frames():
            return
        if 0 <= frame_idx < len(self.frames) and frame_idx != self.current_frame_idx:
            self.current_frame_idx = frame_idx
            self.enhancement.reset_kalman()
            self.update_frame_info()
            self.update_display()
            


    def generate_mask(self):
        if not self.has_frames():
            messagebox.showwarning("Warning", "Please load a TIF file first")
            return
        
        # Create a new toplevel window for the preview
        preview = tk.Toplevel(self.root)
        preview.title("Mask Preview")
        preview.configure(bg='#333333')
        preview.transient(self.root)
        preview.grab_set()
        
        # Get current screen dimensions
        screen_width = preview.winfo_screenwidth()
        screen_height = preview.winfo_screenheight()
        
        # Set the window size to 80% of screen dimensions
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # Prepare the images
        frame = self.current_frame()
        mask = self.current_mask()
        
        # Create mask display (white on black)
        mask_display = np.zeros(frame.shape, dtype=np.uint8)
        mask_display[mask > 0] = [255, 255, 255]
        
        # Create overlay with mask in red
        overlay = frame.copy()
        overlay[mask > 0] = [255, 0, 0]
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, overlay)
        
        # Calculate the aspect ratio of the images
        aspect_ratio = frame.shape[1] / frame.shape[0]
        
        # Calculate the maximum image display size to fit in the window
        # We'll display 3 images side by side, each taking about 1/3 of the window width
        display_width = int(window_width / 3.2)  # Slightly less than 1/3 to account for padding
        display_height = int(display_width / aspect_ratio)
        
        # If calculated height exceeds the available height, recalculate based on height
        max_display_height = int(window_height * 0.8)  # Leave room for buttons and labels
        if display_height > max_display_height:
            display_height = max_display_height
            display_width = int(display_height * aspect_ratio)
        
        # Resize images to the calculated size
        orig_pil = Image.fromarray(frame).resize((display_width, display_height), Image.Resampling.LANCZOS)
        mask_pil = Image.fromarray(mask_display).resize((display_width, display_height), Image.Resampling.LANCZOS)
        overlay_pil = Image.fromarray(overlay).resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage for display
        orig_img = ImageTk.PhotoImage(orig_pil)
        mask_img = ImageTk.PhotoImage(mask_pil)
        overlay_img = ImageTk.PhotoImage(overlay_pil)
        
        # Create the main frame
        main_frame = ttk.Frame(preview)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a frame for the images
        img_frame = ttk.Frame(main_frame)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add the original image
        orig_col = ttk.Frame(img_frame)
        orig_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(orig_col, text="Original", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        ttk.Label(orig_col, image=orig_img).pack()
        
        # Add the mask image
        mask_col = ttk.Frame(img_frame)
        mask_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(mask_col, text="Mask", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        ttk.Label(mask_col, image=mask_img).pack()
        
        # Add the overlay image
        overlay_col = ttk.Frame(img_frame)
        overlay_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(overlay_col, text="Overlay", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        ttk.Label(overlay_col, image=overlay_img).pack()
        
        # Add frame metadata label
        metadata_label = ttk.Label(
            main_frame, 
            text=f"Frame {self.current_frame_idx + 1} of {len(self.frames)} | Image Size: {frame.shape[1]}×{frame.shape[0]} | Mask Pixel Count: {np.sum(mask > 0)}",
            font=("Arial", 10)
        )
        metadata_label.pack(pady=(15, 5))
        
        # Add buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=15)
        
        # Create larger buttons with more padding
        btn_style = {'width': 20, 'padding': 10}
        
        accept_btn = ttk.Button(
            btn_frame, 
            text="Accept & Next (Enter)", 
            command=lambda: self._accept_mask_and_next(preview),
            **btn_style
        )
        accept_btn.pack(side=tk.LEFT, padx=15)
        
        cancel_btn = ttk.Button(
            btn_frame, 
            text="Cancel (Esc)", 
            command=preview.destroy,
            **btn_style
        )
        cancel_btn.pack(side=tk.LEFT, padx=15)
        
        # Add keyboard shortcuts
        preview.bind("<Return>", lambda e: self._accept_mask_and_next(preview))
        preview.bind("<Escape>", lambda e: preview.destroy())
        
        # Store the PhotoImage references to prevent garbage collection
        preview.orig_img = orig_img
        preview.mask_img = mask_img
        preview.overlay_img = overlay_img
        
        # Position the window in the center of the screen
        preview.geometry(f"{window_width}x{window_height}+{int((screen_width - window_width) / 2)}+{int((screen_height - window_height) / 2)}")
        
        # Make the window adjust its size to fit content, but not exceed the specified dimensions
        preview.update_idletasks()
        actual_width = min(window_width, preview.winfo_reqwidth())
        actual_height = min(window_height, preview.winfo_reqheight())
        preview.geometry(f"{actual_width}x{actual_height}")
    def _accept_mask_and_next(self, preview_window):
        result = self.file_manager.save_current_mask()
        preview_window.destroy()
        if result:
            temp_msg = tk.Toplevel(self.root)
            temp_msg.title("Success")
            temp_msg.geometry("300x100")
            temp_msg.configure(bg='#333333')
            temp_msg.resizable(False, False)
            temp_msg.transient(self.root)
            temp_msg.update_idletasks()
            width = temp_msg.winfo_width()
            height = temp_msg.winfo_height()
            x = self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2
            y = self.root.winfo_rooty() + (self.root.winfo_height() - height) // 2
            temp_msg.geometry(f"+{x}+{y}")
            frame_temp = ttk.Frame(temp_msg, padding=15)
            frame_temp.pack(fill=tk.BOTH, expand=True)
            ttk.Label(frame_temp, text="Frame and mask saved successfully", font=("Arial", 10), justify="center").pack(pady=10)
            temp_msg.after(3000, temp_msg.destroy)
        skip = self.navigator.skip_var.get()
        self.next_frame(skip)
        
    def propagate_labels(self, source_frame_idx, target_frame_idx):
        if source_frame_idx < 0 or source_frame_idx >= len(self.frames) or target_frame_idx < 0 or target_frame_idx >= len(self.frames):
            return None
        src_frame = cv2.cvtColor(self.frames[source_frame_idx], cv2.COLOR_RGB2GRAY)
        tgt_frame = cv2.cvtColor(self.frames[target_frame_idx], cv2.COLOR_RGB2GRAY)
        try:
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
            flow = optical_flow.calc(src_frame, tgt_frame, None)
        except:
            flow = cv2.calcOpticalFlowFarneback(src_frame, tgt_frame, None, pyr_scale=0.5, levels=3,
                                                  winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        src_mask = self.masks[source_frame_idx]
        h, w = src_mask.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped_mask = cv2.remap(src_mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return warped_mask
        
    def undo_last_action(self):
        if hasattr(self, 'previous_mask') and self.previous_mask is not None:
            self.masks[self.current_frame_idx] = self.previous_mask.copy()
            self.update_display()
            self.previous_mask = None
        
    def loop_playback(self):
        if not self.has_frames():
            return
        try:
            loop_span = int(self.loop_span_var.get())
        except:
            loop_span = 30
        start_frame = self.current_frame_idx
        end_frame = min(start_frame + loop_span, len(self.frames) - 1)
        frames_to_play = list(range(start_frame, end_frame + 1)) + list(range(end_frame, start_frame - 1, -1))
        def play_loop(idx=0):
            if idx < len(frames_to_play):
                self.set_current_frame(frames_to_play[idx])
                self.root.after(100, lambda: play_loop(idx+1))
            else:
                self.set_current_frame(start_frame)
        play_loop()
        
    def _show_shortcuts(self):
        try:
            shortcuts = tk.Toplevel(self.root)
            shortcuts.title("Keyboard Shortcuts")
            shortcuts.configure(bg='#333333')
            shortcuts.transient(self.root)
            frame = ttk.Frame(shortcuts, padding=20)
            frame.pack(fill=tk.BOTH, expand=True)
            ttk.Label(frame, text="Keyboard Shortcuts", font=("Arial", 16, "bold")).pack(pady=(0,15))
            shortcuts_text = (
                "Left Arrow - Previous frame\n"
                "Right Arrow - Next frame\n"
                "+ (Plus) - Zoom in\n"
                "- (Minus) - Zoom out\n"
                "Ctrl+0 - Reset view\n"
                "c - Center image\n"
                "Ctrl+z - Undo\n"
                "Ctrl+s - Generate mask\n"
                "Ctrl+a - Save all masks\n"
                "Ctrl+i - Intelligent frame selection\n"
                "Ctrl+d - Toggle debug mode\n"
            )
            ttk.Label(frame, text=shortcuts_text, font=("Courier", 12), justify="left").pack(pady=10)
            ttk.Button(frame, text="Close", command=shortcuts.destroy, width=20).pack(pady=(10,0))
            shortcuts.update_idletasks()
            width = shortcuts.winfo_width()
            height = shortcuts.winfo_height()
            x = self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2
            y = self.root.winfo_rooty() + (self.root.winfo_height() - height) // 2
            shortcuts.geometry(f"{width}x{height}+{x}+{y}")
            shortcuts.grab_set()
            shortcuts.focus_set()
        except Exception as e:
            print(f"Error showing shortcuts: {e}")
        
    def _show_about(self):
        try:
            about = tk.Toplevel(self.root)
            about.title("About TIF Labeler")
            about.configure(bg='#333333')
            about.transient(self.root)
            frame = ttk.Frame(about, padding=20)
            frame.pack(fill=tk.BOTH, expand=True)
            ttk.Label(frame, text="TIF Manual Labeling Tool v2.0", font=("Arial", 16, "bold")).pack(pady=(0,15))
            about_text = (
                "A tool for manually labeling regions in TIF image stacks.\n\n"
                "Features:\n"
                "• Multi-frame TIF file support\n"
                "• Intelligent Frame Selection\n"
                "• Zoom and pan functionality\n"
                "• Skip frames for faster labeling\n"
                "• Export individual masks or stacks\n"
                "• Auto-centering and view controls\n\n"
                "©2025 All Rights Reserved"
            )
            ttk.Label(frame, text=about_text, font=("Arial", 11), justify="center").pack(pady=10)
            ttk.Button(frame, text="Close", command=about.destroy, width=20).pack(pady=(10,0))
            about.update_idletasks()
            width = about.winfo_width()
            height = about.winfo_height()
            x = self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2
            y = self.root.winfo_rooty() + (self.root.winfo_height() - height) // 2
            about.geometry(f"{width}x{height}+{x}+{y}")
            about.grab_set()
            about.focus_set()
        except Exception as e:
            print(f"Error showing about dialog: {e}")


def integrate_frame_selector(app):
    from tkinter import Menu
    integration = FrameSelectorIntegration(app)
    view_menu = app.view_menu
    view_menu.add_separator()
    view_menu.add_command(label="Intelligent Frame Selection...", command=integration.show_frame_selector, accelerator="Ctrl+I")
    app.root.bind("<Control-i>", lambda e: integration.show_frame_selector())
    app.frame_selector_integration = integration
    return integration


class FrameSelectorIntegration:
    """Integration class for adding intelligent frame selection to TIFLabeler"""
    def __init__(self, app):
        self.app = app
        
    def add_to_menu(self):
        view_menu = self.app.view_menu
        view_menu.add_separator()
        view_menu.add_command(label="Intelligent Frame Selection...", command=self.show_frame_selector, accelerator="Ctrl+I")
        self.app.root.bind("<Control-i>", lambda e: self.show_frame_selector())
        
    def show_frame_selector(self):
        if not self.app.has_frames():
            messagebox.showerror("Error", "Please load a TIF file first")
            return
        selector_dialog = FrameSelectorDialog(self.app.root, self.app)
        
    def add_to_toolbar(self, parent_frame):
        ttk.Button(parent_frame, text="Auto Select", command=self.show_frame_selector, width=12).pack(side=tk.RIGHT, padx=2)


# Part 5/5: Startup, Main Function, and Application Launch

def setup_drag_drop_support():
    if sys.platform == 'win32':
        try:
            import tkinterdnd2
            return True
        except ImportError:
            try:
                import tkdnd
                return True
            except ImportError:
                return False
    else:
        return True

def show_welcome_message(root):
    welcome = tk.Toplevel(root)
    welcome.title("Welcome to TIF Labeler v2.0")
    welcome.configure(bg='#333333')
    welcome.transient(root)
    welcome.grab_set()
    frame = ttk.Frame(welcome, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)
    title_label = ttk.Label(frame, text="TIF Manual Labeling Tool v2.0", font=("Arial", 16, "bold"))
    title_label.pack(pady=(0,15))
    instructions = (
        "Welcome to the enhanced TIF Labeler!\n\n"
        "New Features:\n"
        "• Intelligent Frame Selection (Ctrl+I) for automatic selection of key frames\n"
        "• Loading animation with frame counter\n"
        "• Automatic window sizing for optimal viewing\n"
        "• Zoom and pan functionality (mouse wheel, middle-click)\n"
        "• Skip frames feature for faster navigation\n"
        "• Frame slider for quick navigation\n"
        "• Cross-platform drag and drop support\n"
        "• Enhanced mask generation and preview\n\n"
        "Quick Start:\n"
        "1. Open a TIF file via File menu or drag-and-drop\n"
        "2. Use Intelligent Frame Selection to identify key frames\n"
        "3. Navigate frames with the slider or buttons\n"
        "4. Draw labels with left-click and erase with the erase tool\n"
        "5. Generate and save masks with the 'Generate Mask' button\n\n"
        "Press OK to begin."
    )
    instructions_label = ttk.Label(frame, text=instructions, font=("Arial", 11), justify="left")
    instructions_label.pack(pady=10)
    ok_button = ttk.Button(frame, text="OK", command=welcome.destroy, width=20)
    ok_button.pack(pady=(10,0))
    welcome.update_idletasks()
    width = welcome.winfo_width()
    height = welcome.winfo_height()
    x = root.winfo_rootx() + (root.winfo_width() - width) // 2
    y = root.winfo_rooty() + (root.winfo_height() - height) // 2
    welcome.geometry(f"{width}x{height}+{x}+{y}")
    welcome.after(15000, welcome.destroy)

def main():
    if setup_drag_drop_support() and sys.platform == 'win32':
        try:
            import tkinterdnd2
            root = tkinterdnd2.TkinterDnD.Tk()
        except:
            root = tk.Tk()
    else:
        root = tk.Tk()
    root.geometry("1200x800")
    root.configure(bg='#333333')
    root.title("Manual Labeling Tool v2.0")
    app = TIFLabeler(root)
    frame_selector_integration = integrate_frame_selector(app)
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.after(500, lambda: show_welcome_message(root))
    root.mainloop()

if __name__ == "__main__":
    main()