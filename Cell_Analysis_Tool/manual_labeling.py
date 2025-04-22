# manual_labeling.py

import sys
import os
import math
import numpy as np
import random
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QFrame, QSizePolicy, QSpinBox,
                           QSlider, QFileDialog, QMessageBox, QDialog, QTextEdit, 
                           QListWidget, QProgressBar)
from PyQt6.QtCore import (Qt, QSize, QPropertyAnimation, QEasingCurve, QTimer,
                        QPointF, pyqtSignal, pyqtSlot, QThread, QObject)
from PyQt6.QtGui import (QIcon, QPixmap, QColor, QPalette, QFont, QPainter, 
                        QBrush, QPen, QImage, QLinearGradient, QRadialGradient)
import cv2
from PIL import Image, ImageQt

# Import from your main application file
from main import StyleSheet, ThemeToggle, ElegantButton, CellVisualization

class ManualLabelingScreen(QWidget):
    """Screen for manual cell labeling with TIF labeler integration"""
    
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.parent_app = parent  # Store reference to parent app
        self.current_file = None
        self.frames = []
        self.masks = []
        self.current_frame_idx = 0
        self.tool = "draw"
        self.brush_size = 5
        self.output_dir = None
        self.previous_mask = None  # For undo feature
        self.file_manager = LabelingFileManager(self)
        self.init_ui()
            
    def init_ui(self):
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Top toolbar
        self.toolbar_frame = QFrame(self)
        self.toolbar_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        toolbar_layout = QHBoxLayout(self.toolbar_frame)
        
        # Back button
        self.back_btn = ElegantButton("Back", self, self.dark_mode, is_primary=False, icon="back")
        self.back_btn.setFixedSize(100, 40)
        self.back_btn.clicked.connect(self.on_back_clicked)
        toolbar_layout.addWidget(self.back_btn)
        
        # Title
        self.title_label = QLabel("Manual Cell Labeling", self.toolbar_frame)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setWeight(QFont.Weight.Medium)
        self.title_label.setFont(title_font)
        toolbar_layout.addWidget(self.title_label)
        toolbar_layout.addStretch()
        
        # Frame counter
        self.frame_counter = QLabel("Frame: 0/0", self.toolbar_frame)
        toolbar_layout.addWidget(self.frame_counter)
        
        # Theme toggle in top right
        self.theme_toggle = ThemeToggle(self, self.dark_mode)
        toolbar_layout.addWidget(self.theme_toggle)
        toolbar_layout.setContentsMargins(10, 10, 20, 10)
        
        layout.addWidget(self.toolbar_frame)
        
        # Main content area (split view - canvas and tools)
        self.content_frame = QFrame(self)
        self.content_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout = QHBoxLayout(self.content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Left side: Canvas area
        self.canvas_frame = QFrame(self.content_frame)
        self.canvas_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas_layout = QVBoxLayout(self.canvas_frame)
        canvas_layout.setContentsMargins(10, 10, 10, 10)

        # Canvas
        self.canvas = CellLabelingCanvas(self, self.dark_mode)
        canvas_layout.addWidget(self.canvas)
        
        content_layout.addWidget(self.canvas_frame, 3)  # 3:1 ratio

        # Right side: Tools panel
        self.tools_panel = QFrame(self.content_frame)
        self.tools_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.tools_panel.setFixedWidth(250)
        tools_layout = QVBoxLayout(self.tools_panel)
        tools_layout.setContentsMargins(15, 15, 15, 15)
        
        # Tools label
        tools_label = QLabel("Tools", self.tools_panel)
        tools_font = QFont()
        tools_font.setPointSize(14)
        tools_font.setWeight(QFont.Weight.Medium)
        tools_label.setFont(tools_font)
        tools_layout.addWidget(tools_label)
        
        # Tool buttons
        self.draw_btn = ElegantButton("Draw", self, self.dark_mode, is_primary=True)
        self.draw_btn.setMinimumHeight(60)
        tools_layout.addWidget(self.draw_btn)
        
        self.erase_btn = ElegantButton("Erase", self, self.dark_mode, is_primary=False)
        self.erase_btn.setMinimumHeight(60)
        tools_layout.addWidget(self.erase_btn)
        
        self.clear_btn = ElegantButton("Clear Mask", self, self.dark_mode, is_primary=False)
        self.clear_btn.setMinimumHeight(60)
        tools_layout.addWidget(self.clear_btn)
        
        # Loop control
        loop_frame = QFrame(self.tools_panel)
        loop_layout = QHBoxLayout(loop_frame)
        loop_layout.setContentsMargins(0, 0, 0, 0)
        
        loop_label = QLabel("Loop Span:", loop_frame)
        loop_layout.addWidget(loop_label)
        
        self.loop_span = QSpinBox(loop_frame)
        self.loop_span.setRange(5, 200)
        self.loop_span.setValue(30)
        loop_layout.addWidget(self.loop_span)
        
        tools_layout.addWidget(loop_frame)
        
        self.loop_btn = ElegantButton("Loop", self, self.dark_mode, is_primary=False)
        self.loop_btn.setMinimumHeight(60)
        self.loop_btn.clicked.connect(self.loop_playback)
        tools_layout.addWidget(self.loop_btn)

        
        self.generate_btn = ElegantButton("Generate Mask", self, self.dark_mode, is_primary=False)
        self.generate_btn.setMinimumHeight(60)
        tools_layout.addWidget(self.generate_btn)
        
        # Brush size slider
        brush_size_frame = QFrame(self.tools_panel)
        brush_layout = QVBoxLayout(brush_size_frame)
        brush_layout.setContentsMargins(0, 10, 0, 10)
        
        brush_label = QLabel("Brush Size:", brush_size_frame)
        brush_layout.addWidget(brush_label)
        
        self.brush_slider = QSlider(Qt.Orientation.Horizontal, brush_size_frame)
        self.brush_slider.setRange(1, 20)
        self.brush_slider.setValue(5)
        brush_layout.addWidget(self.brush_slider)
        
        tools_layout.addWidget(brush_size_frame)
        
        # File open button
        self.open_file_btn = ElegantButton("Open TIF File", self, self.dark_mode, is_primary=True)
        tools_layout.addWidget(self.open_file_btn)
        
        tools_layout.addStretch()
        
        content_layout.addWidget(self.tools_panel, 1)  # 3:1 ratio
        
        layout.addWidget(self.content_frame)
        
        # Bottom navigation bar
        self.nav_bar = QFrame(self)
        self.nav_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        nav_layout = QVBoxLayout(self.nav_bar)
        nav_layout.setContentsMargins(15, 5, 15, 15)
        
        # Navigation controls
        nav_controls = QFrame(self.nav_bar)
        controls_layout = QHBoxLayout(nav_controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        self.prev_btn = QPushButton("Previous", nav_controls)
        controls_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next", nav_controls)
        controls_layout.addWidget(self.next_btn)
        
        controls_layout.addStretch()
        
        skip_label = QLabel("Skip:", nav_controls)
        controls_layout.addWidget(skip_label)
        
        self.skip_frames = QSpinBox(nav_controls)
        self.skip_frames.setRange(1, 100)
        self.skip_frames.setValue(50)
        controls_layout.addWidget(self.skip_frames)
        
        nav_layout.addWidget(nav_controls)
        
        # Frame slider (create before connecting)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal, self.nav_bar)
        nav_layout.addWidget(self.frame_slider)
        
        layout.addWidget(self.nav_bar)
        
        # Connect signals AFTER creating all widgets
        self.draw_btn.clicked.connect(lambda: self.set_tool("draw"))
        self.erase_btn.clicked.connect(lambda: self.set_tool("erase"))
        self.clear_btn.clicked.connect(self.clear_mask)
        
        # Connect navigation buttons
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        
        # Connect file open button
        self.open_file_btn.clicked.connect(self.file_manager.open_file)
        
        # Connect generate button
        self.generate_btn.clicked.connect(self.show_mask_preview)
        
        # Connect sliders
        self.frame_slider.valueChanged.connect(self.set_current_frame)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        
        # Set initial theme
        self.update_theme()
        
    def show_loading_animation(self):
        """Show loading animation before TIF data is loaded"""
        # We'll add the actual loading animation in future code
        pass
        
    def toggle_theme(self, dark_mode):
        """Handle theme toggle from ThemeToggle widget"""
        self.dark_mode = dark_mode
        self.update_theme()
        # Propagate to parent
        if hasattr(self.parent(), 'toggle_theme'):
            self.parent().toggle_theme(dark_mode)

    def show_mask_preview(self):
        """Show mask preview dialog"""
        if not self.frames:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
            
        # Create preview dialog
        preview = QDialog(self)
        preview.setWindowTitle("Mask Preview")
        preview.resize(800, 600)
        
        # Layout
        layout = QVBoxLayout(preview)
        
        # Title
        title = QLabel("Mask Preview", preview)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setWeight(QFont.Weight.Medium)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Images layout
        images_layout = QHBoxLayout()
        
        # Original image
        orig_frame = QLabel(preview)
        orig_frame.setMinimumSize(200, 200)
        orig_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Mask image
        mask_frame = QLabel(preview)
        mask_frame.setMinimumSize(200, 200)
        mask_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Overlay image
        overlay_frame = QLabel(preview)
        overlay_frame.setMinimumSize(200, 200)
        overlay_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add to layout
        orig_container = QVBoxLayout()
        orig_container.addWidget(QLabel("Original", preview))
        orig_container.addWidget(orig_frame)
        
        mask_container = QVBoxLayout()
        mask_container.addWidget(QLabel("Mask", preview))
        mask_container.addWidget(mask_frame)
        
        overlay_container = QVBoxLayout()
        overlay_container.addWidget(QLabel("Overlay", preview))
        overlay_container.addWidget(overlay_frame)
        
        images_layout.addLayout(orig_container)
        images_layout.addLayout(mask_container)
        images_layout.addLayout(overlay_container)
        
        layout.addLayout(images_layout)
        
        # Frame info
        frame = self.frames[self.current_frame_idx]
        mask = self.masks[self.current_frame_idx]
        mask_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage = mask_pixels / total_pixels * 100
        
        info_text = (f"Frame {self.current_frame_idx + 1} of {len(self.frames)} • "
                    f"Size: {frame.shape[1]}×{frame.shape[0]} • "
                    f"Mask pixels: {mask_pixels} • "
                    f"Coverage: {coverage:.1f}%")
        
        info_label = QLabel(info_text, preview)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_btn = ElegantButton("Accept & Next", preview, self.dark_mode, is_primary=True)
        save_btn.clicked.connect(lambda: self.accept_mask_and_next(preview))
        
        cancel_btn = ElegantButton("Cancel", preview, self.dark_mode, is_primary=False)
        cancel_btn.clicked.connect(preview.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Prepare images
        frame_rgb = frame
        
        # Create mask display
        mask_display = np.zeros_like(frame)
        mask_display[mask > 0] = [255, 255, 255]
        
        # Create overlay
        overlay = frame.copy()
        overlay[mask > 0] = [255, 0, 0]
        alpha = 0.3
        overlay_display = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
        
        # Convert to QPixmap and display
        height, width = frame.shape[:2]
        
        # Scale down if too large
        max_display = 300
        scale = min(max_display / width, max_display / height)
        
        # Convert to Qt format
        frame_qimg = QImage(frame_rgb.data, width, height, frame.strides[0], QImage.Format.Format_RGB888)
        mask_qimg = QImage(mask_display.data, width, height, mask_display.strides[0], QImage.Format.Format_RGB888)
        overlay_qimg = QImage(overlay_display.data, width, height, overlay_display.strides[0], QImage.Format.Format_RGB888)
        
        # Create QPixmap and resize
        frame_pixmap = QPixmap.fromImage(frame_qimg).scaled(
            int(width * scale), int(height * scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        mask_pixmap = QPixmap.fromImage(mask_qimg).scaled(
            int(width * scale), int(height * scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        overlay_pixmap = QPixmap.fromImage(overlay_qimg).scaled(
            int(width * scale), int(height * scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Set pixmaps to labels
        orig_frame.setPixmap(frame_pixmap)
        mask_frame.setPixmap(mask_pixmap)
        overlay_frame.setPixmap(overlay_pixmap)
        
        # Apply theme
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        preview.setStyleSheet(f"background-color: {theme['bg_primary']}; color: {theme['text_primary']};")
        
        # Show dialog
        preview.exec()

    
    def update_theme(self):
        """Update UI elements based on the current theme"""
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        # Update backgrounds
        self.setStyleSheet(f"background-color: {theme['bg_primary']};")
        self.toolbar_frame.setStyleSheet(f"background-color: {theme['title_bar']}; border-bottom: 1px solid {theme['border']};")
        self.tools_panel.setStyleSheet(f"background-color: {theme['bg_secondary']}; border-left: 1px solid {theme['border']};")
        self.nav_bar.setStyleSheet(f"background-color: {theme['bg_primary']}; border-top: 1px solid {theme['border']};")
        
        # Update text colors
        self.title_label.setStyleSheet(f"color: {theme['text_primary']};")
        self.frame_counter.setStyleSheet(f"color: {theme['text_secondary']};")
        
        # Update buttons
        self.back_btn.dark_mode = self.dark_mode
        self.back_btn.update_style()
        self.draw_btn.dark_mode = self.dark_mode
        self.draw_btn.update_style()
        self.erase_btn.dark_mode = self.dark_mode
        self.erase_btn.update_style()
        self.clear_btn.dark_mode = self.dark_mode
        self.clear_btn.update_style()
        self.loop_btn.dark_mode = self.dark_mode
        self.loop_btn.update_style()
        self.generate_btn.dark_mode = self.dark_mode
        self.generate_btn.update_style()
        
    def on_back_clicked(self):
        """Go back to the analysis type screen"""
        if self.parent_app is not None and hasattr(self.parent_app, 'show_analysis_type_screen'):
            self.parent_app.show_analysis_type_screen()

    def update_frame_info(self):
        """Update frame counter and slider position"""
        total = len(self.frames) if self.frames else 0
        if total > 0:
            self.frame_counter.setText(f"Frame: {self.current_frame_idx + 1}/{total}")
            # Update slider
            self.frame_slider.setMaximum(total - 1)
            self.frame_slider.setValue(self.current_frame_idx)
        else:
            self.frame_counter.setText("Frame: 0/0")
        
    def set_tool(self, tool):
        """Set the current drawing tool"""
        self.tool = tool
        
    def update_brush_size(self, size):
        """Update brush size from slider"""
        self.brush_size = size
        
    def set_current_frame(self, idx):
        """Set the current frame index and update display"""
        if not self.frames:
            return
            
        if 0 <= idx < len(self.frames):
            self.current_frame_idx = idx
            self.update_frame_info()
            self.canvas.set_frame_and_mask(
                self.frames[idx],
                self.masks[idx]
            )
            
    def prev_frame(self):
        """Go to previous frame"""
        if self.current_frame_idx > 0:
            self.set_current_frame(self.current_frame_idx - 1)
            
    def next_frame(self):
        """Go to next frame with optional skip"""
        skip = self.skip_frames.value()
        next_idx = min(self.current_frame_idx + skip, len(self.frames) - 1)
        self.set_current_frame(next_idx)
        
    def clear_mask(self):
        """Clear the current mask"""
        if not self.frames:
            return
            
        # Save for undo
        self.previous_mask = self.masks[self.current_frame_idx].copy()
        
        # Clear mask
        self.masks[self.current_frame_idx].fill(0)
        
        # Update display
        self.canvas.set_frame_and_mask(
            self.frames[self.current_frame_idx],
            self.masks[self.current_frame_idx]
        )
        
    def undo_last_action(self):
        """Undo the last drawing action"""
        if self.previous_mask is not None:
            # Swap current and previous
            current = self.masks[self.current_frame_idx].copy()
            self.masks[self.current_frame_idx] = self.previous_mask
            self.previous_mask = current
            
            # Update display
            self.canvas.set_frame_and_mask(
                self.frames[self.current_frame_idx],
                self.masks[self.current_frame_idx]
            )


    def loop_playback(self):
        """Play a loop of frames around current frame"""
        if not self.frames:
            return
            
        # Get loop span
        span = self.loop_span.value()
        
        # Calculate start and end frames
        start_frame = max(0, self.current_frame_idx - span // 2)
        end_frame = min(len(self.frames) - 1, self.current_frame_idx + span // 2)
        
        # Create frames to play (forward and backward)
        frames_to_play = list(range(start_frame, end_frame + 1)) + list(range(end_frame, start_frame - 1, -1))
        
        # Disable UI during playback
        self.setEnabled(False)
        
        # Create playback timer
        play_timer = QTimer(self)
        current_idx = 0
        
        def play_next():
            nonlocal current_idx
            if current_idx < len(frames_to_play):
                self.set_current_frame(frames_to_play[current_idx])
                current_idx += 1
            else:
                # End of playback
                play_timer.stop()
                self.setEnabled(True)
                
        play_timer.timeout.connect(play_next)
        play_timer.start(100)  # 10 fps


    def show_frame_selector(self):
        """Show the frame selector dialog"""
        if not self.frames:
            QMessageBox.warning(self, "Warning", "No frames loaded. Please load a TIF file first.")
            return
            
        dialog = FrameSelectorDialog(self, self.dark_mode)
        dialog.exec()
        
    def connect_buttons(self):
        """Connect all buttons to their functions"""
        # Connect previously defined buttons
        self.draw_btn.clicked.connect(lambda: self.set_tool("draw"))
        self.erase_btn.clicked.connect(lambda: self.set_tool("erase"))
        self.clear_btn.clicked.connect(self.clear_mask)
        
        # Connect navigation buttons
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        
        # Connect slider
        self.frame_slider.valueChanged.connect(self.set_current_frame)
        
        # Connect brush size slider
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        
        # Add file open button to tools panel
        self.open_file_btn = ElegantButton("Open TIF File", self, self.dark_mode, is_primary=True)
        self.open_file_btn.clicked.connect(self.file_manager.open_file)
        
        # Connect generate button
        self.generate_btn.clicked.connect(self.show_mask_preview)
        
        # Connect loop button
        self.loop_btn.clicked.connect(self.loop_playback)
        
        # Add frame selector button
        self.frame_selector_btn = ElegantButton("Select Key Frames", self, self.dark_mode, is_primary=False)
        self.frame_selector_btn.clicked.connect(self.show_frame_selector)

    def resize_window_to_image(self, frame):
        """Resize the window to fit the image better"""
        if frame is None:
            return
            
        # Get image dimensions
        height, width = frame.shape[:2]
        
        # Get screen dimensions
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()
        
        # Calculate scaling to fit within 80% of screen
        max_width = int(screen_width * 0.8)
        max_height = int(screen_height * 0.8)
        
        # Calculate default zoom
        ratio = min(max_width / width, max_height / height)
        
        # Update canvas zoom
        self.canvas.zoom_factor = ratio
        
        # Resize window
        window_width = min(max_width, int(width * ratio) + self.tools_panel.width() + 50)
        window_height = min(max_height, int(height * ratio) + self.toolbar_frame.height() + self.nav_bar.height() + 50)
        
        self.parent_app.resize(window_width, window_height)


    def accept_mask_and_next(self, dialog):
        """Save current mask and move to next frame"""
        try:
            # Ensure frames and masks exist
            if not self.frames or not self.masks:
                QMessageBox.warning(self, "Warning", "No frames loaded")
                return
            
            # Save mask using file manager
            if hasattr(self, 'file_manager'):
                result = self.file_manager.save_current_mask()
                
                if result:
                    # Close dialog
                    dialog.accept()
                    
                    # Show success message
                    QMessageBox.information(self, "Success", "Frame and mask saved successfully")
                    
                    # Move to next frame
                    skip = self.skip_frames.value()
                    next_idx = min(self.current_frame_idx + skip, len(self.frames) - 1)
                    self.set_current_frame(next_idx)
            else:
                QMessageBox.warning(self, "Error", "File manager not initialized")
        
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to save mask and move to next frame:\n{str(e)}"
            )

    def save_current_mask(self):
        """Wrapper method to save current mask"""
        if hasattr(self, 'file_manager'):
            return self.file_manager.save_current_mask()
        else:
            QMessageBox.warning(self, "Error", "File manager not initialized")
            return False
    


class LabelingFileManager:
    """Handles file operations for the labeling workflow"""
    
    def __init__(self, parent):
        self.parent = parent
        self.output_dir = None
        

    def load_file(self, file_path):
        """Load a TIF file with comprehensive error handling and multiple loading methods"""
        try:
            # Try OpenCV first
            frames = []
            masks = []
            
            # Method 1: OpenCV VideoCapture
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    for i in range(frame_count):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        
                        # Create empty mask
                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        masks.append(mask)
                    
                    cap.release()
                    
                    if frames:
                        return frames, masks
            except Exception as cv2_error:
                print(f"OpenCV loading failed: {cv2_error}")
            
            # Method 2: Pillow (PIL)
            try:
                from PIL import Image
                
                # Open the TIFF
                with Image.open(file_path) as img:
                    # Check if it's a multi-page TIFF
                    frames = []
                    masks = []
                    
                    try:
                        while True:
                            # Convert to RGB
                            rgb_frame = img.convert('RGB')
                            
                            # Convert to numpy array
                            frame = np.array(rgb_frame)
                            frames.append(frame)
                            
                            # Create empty mask
                            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                            masks.append(mask)
                            
                            # Move to next frame
                            img.seek(img.tell() + 1)
                    except EOFError:
                        # End of multipage TIFF
                        pass
                    
                    if frames:
                        return frames, masks
            except Exception as pil_error:
                print(f"PIL loading failed: {pil_error}")
            
            # Method 3: scikit-image (if available)
            try:
                import skimage.io
                
                # Read TIFF stack
                tiff_stack = skimage.io.imread(file_path)
                
                if len(tiff_stack.shape) == 3:
                    # Single color channel
                    frames = [tiff_stack[i] for i in range(tiff_stack.shape[0])]
                    masks = [np.zeros(frame.shape[:2], dtype=np.uint8) for frame in frames]
                elif len(tiff_stack.shape) == 4:
                    # RGB or RGBA
                    frames = [tiff_stack[i] for i in range(tiff_stack.shape[0])]
                    masks = [np.zeros(frame.shape[:2], dtype=np.uint8) for frame in frames]
                
                if frames:
                    return frames, masks
            except Exception as skimage_error:
                print(f"scikit-image loading failed: {skimage_error}")
            
            # If all methods fail
            QMessageBox.critical(
                self.parent, 
                "Error", 
                "Failed to load TIF file. Possible reasons:\n"
                "- Unsupported file format\n"
                "- Corrupted or incompatible TIFF\n"
                "- No image data found"
            )
            return [], []
        
        except Exception as e:
            QMessageBox.critical(
                self.parent, 
                "Unexpected Error", 
                f"An unexpected error occurred while loading the file:\n{str(e)}"
            )
            return [], []

    def open_file(self):
        """Open a TIF file dialog with enhanced error handling"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Select TIF file",
            "",
            "TIF files (*.tif *.tiff);;All image files (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)"
        )
        
        if file_path:
            # Load frames and masks
            frames, masks = self.load_file(file_path)
            
            if frames:
                # Update parent with loaded data
                self.parent.current_file = file_path
                self.parent.frames = frames
                self.parent.masks = masks
                self.parent.current_frame_idx = 0
                
                # Update UI
                self.parent.update_frame_info()
                
                # Set initial frame and mask
                self.parent.canvas.set_frame_and_mask(
                    self.parent.frames[0], 
                    self.parent.masks[0]
                )
                
                # Optionally resize window to fit image
                if hasattr(self.parent, 'resize_window_to_image'):
                    self.parent.resize_window_to_image(frames[0])
                
                return True
            else:
                QMessageBox.warning(
                    self.parent, 
                    "No Frames", 
                    "No frames could be extracted from the selected file."
                )
                return False
            
    def set_output_folder(self):
        """Set output folder for masks"""
        folder = QFileDialog.getExistingDirectory(
            self.parent,
            "Select Output Folder for Masks"
        )
        
        if folder:
            self.output_dir = folder
            return True
        return False
        
    def save_current_mask(self):
        """Save the current mask"""
        if not self.parent.current_file or not hasattr(self.parent, 'current_frame_idx'):
            QMessageBox.warning(self.parent, "Warning", "No file loaded")
            return False
            
        if not self.output_dir:
            if not self.set_output_folder():
                QMessageBox.warning(self.parent, "Warning", "No output directory selected")
                return False
                
        try:
            # Get current frame and mask
            frame = self.parent.frames[self.parent.current_frame_idx]
            mask = self.parent.masks[self.parent.current_frame_idx]
            
            # Create filename based on original file
            base_filename = os.path.splitext(os.path.basename(self.parent.current_file))[0]
            frame_filename = f"{base_filename}_frame_{self.parent.current_frame_idx}.tif"
            mask_filename = f"{base_filename}_mask_{self.parent.current_frame_idx}.tif"
            
            # Create subdirectories if they don't exist
            os.makedirs(base_filename, exist_ok=True)
            frames_dir = os.path.join(self.output_dir, "Images")
            masks_dir = os.path.join(self.output_dir, "Masks")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            
            # Save frame and mask
            cv2.imwrite(os.path.join(frames_dir, frame_filename), 
                       cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(masks_dir, mask_filename), mask)
            
            return True
            
        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to save mask: {str(e)}")
            return False
            
    def save_all_masks(self):
        """Save all frames and masks"""
        if not self.parent.frames:
            QMessageBox.warning(self.parent, "Warning", "No frames loaded")
            return False
            
        if not self.output_dir:
            if not self.set_output_folder():
                QMessageBox.warning(self.parent, "Warning", "No output directory selected")
                return False
                
        try:
            # Create subdirectories
            frames_dir = os.path.join(self.output_dir, "Frames")
            masks_dir = os.path.join(self.output_dir, "Masks")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            
            # Create filename based on original file
            base_filename = os.path.splitext(os.path.basename(self.parent.current_file))[0]
            
            # Save each frame and mask
            for i, (frame, mask) in enumerate(zip(self.parent.frames, self.parent.masks)):
                frame_filename = f"{base_filename}_frame_{i}.tif"
                mask_filename = f"{base_filename}_mask_{i}.tif"
                
                cv2.imwrite(os.path.join(frames_dir, frame_filename),
                           cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(masks_dir, mask_filename), mask)
                
            # Also save stacks (all frames and masks in one file)
            stack_frames_path = os.path.join(self.output_dir, f"{base_filename}_frames_stack.tif")
            stack_masks_path = os.path.join(self.output_dir, f"{base_filename}_masks_stack.tif")
            
            # We'll use PIL for multi-page TIF
            pil_frames = [Image.fromarray(f) for f in self.parent.frames]
            pil_masks = [Image.fromarray(m) for m in self.parent.masks]
            
            if pil_frames:
                pil_frames[0].save(stack_frames_path, save_all=True, 
                                  append_images=pil_frames[1:])
            
            if pil_masks:
                pil_masks[0].save(stack_masks_path, save_all=True,
                                 append_images=pil_masks[1:])
                
            QMessageBox.information(
                self.parent, 
                "Success", 
                f"All masks saved to:\n{self.output_dir}\n\nTotal frames: {len(self.parent.frames)}"
            )
            return True
            
        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to save all masks: {str(e)}")
            return False
        


    def show_loading_animation(self):
        """Show loading animation when loading TIF files"""
        # Create loading dialog
        loading_dialog = QDialog(self)
        loading_dialog.setWindowTitle("Loading")
        loading_dialog.setModal(True)
        loading_dialog.resize(400, 400)
        
        # Disable close button
        loading_dialog.setWindowFlags(loading_dialog.windowFlags() & ~Qt.WindowCloseButtonHint)
        
        # Layout
        layout = QVBoxLayout(loading_dialog)
        
        # Add cell animation
        animation = LoadingAnimation(loading_dialog, self.dark_mode)
        layout.addWidget(animation)
        
        # Show dialog without blocking
        loading_dialog.show()
        QApplication.processEvents()
        
        return loading_dialog
    


  
class LoadingAnimation(QWidget):
    """Cell-themed loading animation"""
    
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.setMinimumSize(300, 300)
        self.progress = 0
        self.cells = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)  # Update every 50ms
        self.initialize_cells()
        
    def initialize_cells(self):
        """Create cell animation elements"""
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        # Main central cell
        center = QPointF(self.width()/2, self.height()/2)
        self.main_cell = {
            'position': center,
            'radius': 70,
            'color': QColor(theme['accent']),
            'phase': 0
        }
        
        # Orbital cells
        for i in range(4):
            angle = i * (2 * math.pi / 4)
            self.cells.append({
                'position': QPointF(
                    center.x() + 100 * math.cos(angle),
                    center.y() + 100 * math.sin(angle)
                ),
                'radius': 15 + random.random() * 10,
                'color': QColor(theme['accent_secondary']),
                'phase': random.random() * 2 * math.pi,
                'orbit_speed': 0.001 + random.random() * 0.002,
                'orbit_radius': 100 + random.random() * 20
            })
            
    def animate(self):
        """Update animation state"""
        self.progress += 1
        if self.progress >= 100:
            self.progress = 100
            
        # Update cell animations
        center = QPointF(self.width()/2, self.height()/2)
        for cell in self.cells:
            cell['phase'] += cell['orbit_speed']
            cell['position'] = QPointF(
                center.x() + cell['orbit_radius'] * math.cos(cell['phase']),
                center.y() + cell['orbit_radius'] * math.sin(cell['phase'])
            )
        
        self.update()
        
    def paintEvent(self, event):
        """Draw the loading animation"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        painter.fillRect(self.rect(), QColor(theme['bg_primary']))
        
        # Draw cells
        for cell in self.cells:
            # Draw cell with gradient
            gradient = QRadialGradient(cell['position'], cell['radius'])
            gradient.setColorAt(0, cell['color'])
            gradient.setColorAt(1, cell['color'].darker(150))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(cell['position'], cell['radius'], cell['radius'])
            
        # Draw main cell
        gradient = QRadialGradient(self.main_cell['position'], self.main_cell['radius'])
        gradient.setColorAt(0, self.main_cell['color'])
        gradient.setColorAt(1, self.main_cell['color'].darker(150))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self.main_cell['position'], self.main_cell['radius'], self.main_cell['radius'])
        
        # Draw nucleus
        nucleus_radius = self.main_cell['radius'] * 0.4
        nucleus_pos = QPointF(
            self.main_cell['position'].x() + self.main_cell['radius'] * 0.1,
            self.main_cell['position'].y() - self.main_cell['radius'] * 0.1
        )
        nucleus_color = self.main_cell['color'].darker(200)
        painter.setBrush(QBrush(nucleus_color))
        painter.drawEllipse(nucleus_pos, nucleus_radius, nucleus_radius)
        
        # Draw progress text
        painter.setPen(QColor(theme['text_primary']))
        font = QFont()
        font.setPointSize(14)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                       f"Loading... {self.progress}%")
        
        # Draw progress bar
        bar_width = self.width() * 0.6
        bar_height = 8
        bar_x = (self.width() - bar_width) / 2
        bar_y = self.height() - 40
        
        # Background
        painter.setBrush(QBrush(QColor(theme['bg_secondary'])))
        painter.setPen(QPen(QColor(theme['border']), 1))
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 4, 4)
        
        # Fill
        progress_width = bar_width * (self.progress / 100)
        painter.setBrush(QBrush(QColor(theme['accent'])))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(bar_x, bar_y, progress_width, bar_height, 4, 4)

class CellLabelingCanvas(QWidget):
    """Canvas for cell mask drawing and visualization"""
    
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.parent_app = parent
        self.dark_mode = dark_mode
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drawing = False
        self.panning = False
        self.last_x, self.last_y = 0, 0
        self.current_frame = None
        self.current_mask = None
        self.display_image = None
        self.setMinimumSize(300, 300)
        self.setup_bindings()
        
    def setup_bindings(self):
        """Setup mouse events"""
        self.setMouseTracking(True)
        
    def paintEvent(self, event):
        """Draw the current frame and mask overlay"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        painter.fillRect(self.rect(), QColor(theme["bg_primary"]))
        
        if self.display_image:
            # Calculate position with zoom and pan
            width = self.display_image.width() * self.zoom_factor
            height = self.display_image.height() * self.zoom_factor
            
            # Center image
            x_pos = max(0, (self.width() - width) // 2) + self.pan_x
            y_pos = max(0, (self.height() - height) // 2) + self.pan_y
            
            # Draw the image
            painter.drawImage(QPointF(x_pos, y_pos), 
                            self.display_image.scaled(int(width), int(height), 
                                                    Qt.AspectRatioMode.KeepAspectRatio, 
                                                    Qt.TransformationMode.SmoothTransformation))
        else:
            # Draw a placeholder or loading indicator
            painter.setPen(QColor(theme["text_secondary"]))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
            
    def mousePressEvent(self, event):
        """Handle mouse press for drawing or erasing"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_x, self.last_y = event.pos().x(), event.pos().y()
            
            # Verify parent and mask availability
            if hasattr(self.parent_app, 'masks') and self.parent_app.masks:
                # Save current mask for undo
                self.parent_app.previous_mask = self.parent_app.masks[self.parent_app.current_frame_idx].copy()
            
            self.mouseMoveEvent(event)  # Start drawing immediately
        elif event.button() == Qt.MouseButton.RightButton:
            # Start panning
            self.panning = True
            self.last_pan_x = event.pos().x()
            self.last_pan_y = event.pos().y()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
    def mouseMoveEvent(self, event):
        """Handle mouse movement for drawing, erasing, or panning"""
        if self.drawing and hasattr(self.parent_app, 'tool'):
            # Get image coordinates
            x, y = self._canvas_to_image_coords(event.pos().x(), event.pos().y())
            
            # Verify parent and mask availability
            if hasattr(self.parent_app, 'masks') and self.parent_app.masks:
                mask = self.parent_app.masks[self.parent_app.current_frame_idx]
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    # Draw or erase based on the current tool
                    if self.parent_app.tool == "draw":
                        cv2.circle(mask, (x, y), self.parent_app.brush_size, 255, -1)
                    else:  # Erase
                        cv2.circle(mask, (x, y), self.parent_app.brush_size, 0, -1)
                    
                    # Update display
                    self.update_display()
        
        elif hasattr(self, 'panning') and self.panning:
            # Panning the view
            dx = event.pos().x() - self.last_pan_x
            dy = event.pos().y() - self.last_pan_y
            self.pan_x += dx
            self.pan_y += dy
            self.last_pan_x = event.pos().x()
            self.last_pan_y = event.pos().y()
            self.update()
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        delta = event.angleDelta().y()
        if delta > 0:
            # Zoom in
            self.zoom_factor = min(self.zoom_factor * 1.2, 10.0)
        else:
            # Zoom out
            self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
        self.update()
        
    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates"""
        if self.current_frame is None:
            return 0, 0
            
        # Calculate image position on canvas
        img_w = self.current_frame.shape[1]
        img_h = self.current_frame.shape[0]
        display_w = img_w * self.zoom_factor
        display_h = img_h * self.zoom_factor
        x_pos = max(0, (self.width() - display_w) // 2) + self.pan_x
        y_pos = max(0, (self.height() - display_h) // 2) + self.pan_y
        
        # Check if click is within image bounds
        if (x_pos <= canvas_x <= x_pos + display_w and 
            y_pos <= canvas_y <= y_pos + display_h):
            # Convert to image coordinates
            rel_x = (canvas_x - x_pos) / display_w
            rel_y = (canvas_y - y_pos) / display_h
            image_x = int(rel_x * img_w)
            image_y = int(rel_y * img_h)
            return image_x, image_y
        
        return 0, 0
        
    def update_display(self):
        """Update the display with the current frame and mask overlay"""
        if self.current_frame is not None and self.current_mask is not None:
            # Create overlay
            overlay = self.current_frame.copy()
            # Add red mask overlay
            overlay[self.current_mask > 0] = [255, 0, 0]
            # Blend with original frame
            alpha = 0.3
            display = cv2.addWeighted(overlay, alpha, self.current_frame, 1-alpha, 0)
            
            # Convert to QImage
            height, width, channels = display.shape
            bytes_per_line = channels * width
            qt_image = QImage(display.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.display_image = qt_image
            self.update()
            
    def set_frame_and_mask(self, frame, mask):
        """Set the current frame and mask"""
        self.current_frame = frame
        self.current_mask = mask
        self.update_display()
        
    def reset_view(self):
        """Reset zoom and pan to default"""
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update()


class IntelligentFrameSelector(QObject):
    """Frame selection to identify key frames for labeling"""
    
    # Define signals
    progress_updated = pyqtSignal(int, str)
    selection_complete = pyqtSignal(list)
    selection_error = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.frames = []
        self.error_threshold = 0.15
        self.min_frames = 2
        self.max_frames = None
        self.scene_changes = []
        self.frame_scores = {}
        
    def compute_optical_flow(self, frame1, frame2):
        """Compute optical flow between two frames"""
        # Convert to grayscale
        if len(frame1.shape) == 3:
            f1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        else:
            f1 = frame1
            
        if len(frame2.shape) == 3:
            f2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            f2 = frame2
            
        try:
            # Try DualTVL1 method
            flow = cv2.optflow.DualTVL1OpticalFlow_create().calc(f1, f2, None)
        except:
            # Fall back to Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                f1, f2, None, 
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
        return flow
        
    def calculate_flow_statistics(self, flow):
        """Calculate statistics from optical flow"""
        # Magnitude
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Flow derivatives
        du_dx = cv2.Sobel(flow[..., 0], cv2.CV_64F, 1, 0, ksize=3)
        dv_dy = cv2.Sobel(flow[..., 1], cv2.CV_64F, 0, 1, ksize=3)
        flow_div = du_dx + dv_dy
        
        dv_dx = cv2.Sobel(flow[..., 1], cv2.CV_64F, 1, 0, ksize=3)
        du_dy = cv2.Sobel(flow[..., 0], cv2.CV_64F, 0, 1, ksize=3)
        flow_curl = dv_dx - du_dy
        
        # Return statistics
        stats = {
            'mean_mag': np.mean(flow_mag),
            'max_mag': np.max(flow_mag),
            'p95_mag': np.percentile(flow_mag, 95),
            'mean_div_abs': np.mean(np.abs(flow_div)),
            'mean_curl_abs': np.mean(np.abs(flow_curl)),
            'large_motion_ratio': np.mean(flow_mag > 10.0),
        }
        
        return stats
        
    def estimate_propagation_error(self, frame_idx, labeled_frames=None):
        """Estimate how well a frame can be propagated from labeled frames"""
        if labeled_frames is None or len(labeled_frames) == 0:
            n_frames = len(self.frames)
            return abs(frame_idx - n_frames // 2) / n_frames
            
        # Find nearest labeled frame
        nearest_labeled = min(labeled_frames, key=lambda x: abs(x - frame_idx))
        distance = abs(nearest_labeled - frame_idx)
        
        # If close enough, compute direct optical flow
        max_flow_distance = 5
        if distance <= max_flow_distance:
            frame1 = self.frames[nearest_labeled]
            frame2 = self.frames[frame_idx]
            flow = self.compute_optical_flow(frame1, frame2)
            flow_stats = self.calculate_flow_statistics(flow)
        else:
            # For distant frames, accumulate flow statistics
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
                flow = self.compute_optical_flow(self.frames[idx1], self.frames[idx2])
                stats = self.calculate_flow_statistics(flow)
                
                for key in flow_stats_cumulative:
                    flow_stats_cumulative[key] += stats[key]
                    
            # Average the accumulated stats
            flow_stats = flow_stats_cumulative
            for key in flow_stats:
                flow_stats[key] /= distance
                
        # Compute error score based on flow statistics
        error_score = (
            0.4 * flow_stats['mean_mag'] +
            0.2 * flow_stats['p95_mag'] / 50.0 +
            0.2 * flow_stats['mean_div_abs'] +
            0.1 * flow_stats['mean_curl_abs'] +
            0.1 * flow_stats['large_motion_ratio'] * 10.0
        )
        
        # Scale by distance
        error_score *= np.sqrt(distance)
        
        return error_score
        
    def detect_scene_changes(self):
        """Detect scene changes in video"""
        n_frames = len(self.frames)
        scene_changes = []
        diffs = []
        
        self.progress_updated.emit(10, f"Detecting scene changes in {n_frames} frames...")
        
        for i in range(1, n_frames):
            prev_frame = cv2.cvtColor(self.frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_frame = cv2.cvtColor(self.frames[i], cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(prev_frame, curr_frame)
            diff_mean = np.mean(diff)
            diffs.append(diff_mean)
            
            if i % 10 == 0:
                self.progress_updated.emit(10 + (i * 10) // n_frames, f"Processing frame {i}/{n_frames}...")
        
        # Smooth diffs if we have enough frames
        threshold = 40.0  # Default threshold
        
        if len(diffs) > 5:
            try:
                from scipy.signal import savgol_filter
                diffs_smooth = savgol_filter(diffs, 5, 2)
            except:
                diffs_smooth = diffs
        else:
            diffs_smooth = diffs
            
        # Detect peaks above threshold
        for i in range(1, len(diffs_smooth) - 1):
            if diffs_smooth[i] > diffs_smooth[i-1] and diffs_smooth[i] > diffs_smooth[i+1]:
                if diffs_smooth[i] > threshold:
                    scene_changes.append(i + 1)
                    
        self.progress_updated.emit(20, f"Found {len(scene_changes)} scene changes")
        return scene_changes
        
    def select_frames(self):
        """Main method to select frames"""
        try:
            n_frames = len(self.frames)
            if n_frames == 0:
                self.selection_error.emit("No frames available for selection")
                return []
                
            self.progress_updated.emit(0, f"Starting analysis of {n_frames} frames...")
            
            # Detect scene changes
            self.scene_changes = self.detect_scene_changes()
            
            # Determine budget (number of frames to select)
            if self.max_frames is None:
                budget = max(self.min_frames, int(np.sqrt(n_frames) * 0.5))
            else:
                budget = min(self.max_frames, n_frames)
                
            budget = min(budget, 50)  # Cap at 50 frames for performance
            
            self.progress_updated.emit(30, f"Planning to select {budget} frames...")
            
            # Always select first frame
            selected_frames = [0]
            labeled_frames = [0]
            
            # Calculate error estimates
            error_estimates = [0]  # First frame has 0 error
            
            for j in range(1, n_frames):
                if j % 10 == 0:
                    self.progress_updated.emit(
                        30 + (j * 30) // n_frames, 
                        f"Estimating propagation error for frame {j}/{n_frames}..."
                    )
                    
                err = self.estimate_propagation_error(j, labeled_frames)
                error_estimates.append(err)
                
            # Set adaptive threshold based on error distribution
            mean_err = np.mean(error_estimates)
            std_err = np.std(error_estimates)
            adaptive_threshold = mean_err + 0.5 * std_err
            
            self.progress_updated.emit(60, f"Adaptive threshold set to {adaptive_threshold:.2f}")
            
            # Select frames based on error
            for j in range(1, n_frames):
                if j % 10 == 0:
                    self.progress_updated.emit(
                        60 + (j * 30) // n_frames, 
                        f"Selecting frames... {j}/{n_frames}"
                    )
                    
                if error_estimates[j] > adaptive_threshold:
                    selected_frames.append(j)
                    labeled_frames.append(j)
                    
                    # Update error estimates for all remaining frames
                    for k in range(j+1, n_frames):
                        error_estimates[k] = self.estimate_propagation_error(k, labeled_frames)
            
            # Ensure we don't have too many frames
            if len(selected_frames) > budget:
                # Sample uniformly if we have too many
                step = max(1, len(selected_frames) // budget)
                selected_frames = selected_frames[::step]
                
            # Make sure frames are sorted and unique
            selected_frames = sorted(list(set(selected_frames)))
            
            # Store scores for display
            self.frame_scores = {frame: error_estimates[frame] for frame in selected_frames}
            
            self.progress_updated.emit(100, f"Selection complete: {len(selected_frames)} frames")
            self.selection_complete.emit(selected_frames)
            
            return selected_frames
            
        except Exception as e:
            import traceback
            error_msg = f"Error during frame selection: {str(e)}\n{traceback.format_exc()}"
            self.selection_error.emit(error_msg)
            return []
        

class FrameSelectorDialog(QDialog):
    """Dialog for intelligent frame selection"""
    
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.parent_app = parent
        self.dark_mode = dark_mode
        self.selected_frames = []
        self.selection_in_progress = False
        self.setWindowTitle("Intelligent Frame Selection")
        self.resize(800, 600)
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Intelligent Frame Selection", self)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setWeight(QFont.Weight.Medium)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "This tool automatically selects the most informative frames for manual labeling. "
            "It analyzes video content to identify frames that are most different from each other.",
            self
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)
        
        # Settings frame (shown before selection starts)
        self.settings_frame = QFrame(self)
        settings_layout = QVBoxLayout(self.settings_frame)
        
        # Settings group
        settings_group = QFrame(self.settings_frame)
        settings_group.setFrameShape(QFrame.Shape.StyledPanel)
        group_layout = QVBoxLayout(settings_group)
        
        # Max frames control
        max_frames_layout = QHBoxLayout()
        max_frames_label = QLabel("Maximum frames to select:", self.settings_frame)
        self.max_frames_spin = QSpinBox(self.settings_frame)
        self.max_frames_spin.setRange(2, 100)
        self.max_frames_spin.setValue(10)
        self.max_frames_spin.setEnabled(False)
        
        self.auto_frames_check = QPushButton("Auto (adaptive)", self.settings_frame)
        self.auto_frames_check.setCheckable(True)
        self.auto_frames_check.setChecked(True)
        self.auto_frames_check.clicked.connect(self.toggle_max_frames)
        
        max_frames_layout.addWidget(max_frames_label)
        max_frames_layout.addWidget(self.max_frames_spin)
        max_frames_layout.addWidget(self.auto_frames_check)
        group_layout.addLayout(max_frames_layout)
        
        # Threshold control
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Error threshold:", self.settings_frame)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal, self.settings_frame)
        self.threshold_slider.setRange(5, 25)
        self.threshold_slider.setValue(15)
        threshold_value = QLabel("0.15", self.settings_frame)
        
        def update_threshold_label(value):
            threshold_value.setText(f"{value/100:.2f}")
            
        self.threshold_slider.valueChanged.connect(update_threshold_label)
        
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(threshold_value)
        group_layout.addLayout(threshold_layout)
        
        settings_layout.addWidget(settings_group)
        
        # Start button
        start_btn = ElegantButton("Start Selection", self.settings_frame, self.dark_mode, is_primary=True)
        start_btn.clicked.connect(self.start_selection)
        settings_layout.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.settings_frame)
        
        # Progress frame (shown during selection)
        self.progress_frame = QFrame(self)
        self.progress_frame.setVisible(False)
        progress_layout = QVBoxLayout(self.progress_frame)
        
        self.progress_label = QLabel("Analyzing video...", self.progress_frame)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar(self.progress_frame)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("", self.progress_frame)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        # Log text area
        self.log_text = QTextEdit(self.progress_frame)
        self.log_text.setReadOnly(True)
        progress_layout.addWidget(self.log_text)
        
        layout.addWidget(self.progress_frame)
        
        # Results frame (shown after selection)
        self.results_frame = QFrame(self)
        self.results_frame.setVisible(False)
        results_layout = QHBoxLayout(self.results_frame)
        
        # Left panel - Selected frames list
        left_panel = QFrame(self.results_frame)
        left_layout = QVBoxLayout(left_panel)
        
        left_title = QLabel("Selected Key Frames", left_panel)
        left_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_layout.addWidget(left_title)
        
        self.frames_list = QListWidget(left_panel)
        left_layout.addWidget(self.frames_list)
        
        # Connect list selection
        self.frames_list.itemClicked.connect(self.on_frame_selected)
        
        results_layout.addWidget(left_panel)
        
        # Right panel - Navigation and analysis
        right_panel = QFrame(self.results_frame)
        right_layout = QVBoxLayout(right_panel)
        
        nav_title = QLabel("Navigation", right_panel)
        nav_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(nav_title)
        
        # Navigation controls
        nav_frame = QFrame(right_panel)
        nav_layout = QVBoxLayout(nav_frame)
        
        # Frame position indicator
        self.frame_pos_label = QLabel("Current Position: Frame 0/0", nav_frame)
        nav_layout.addWidget(self.frame_pos_label)
        
        # Frame slider
        self.frame_pos_slider = QSlider(Qt.Orientation.Horizontal, nav_frame)
        self.frame_pos_slider.setEnabled(False)  # Disabled until frames are selected
        nav_layout.addWidget(self.frame_pos_slider)
        
        # Navigation buttons
        buttons_layout = QHBoxLayout()
        self.prev_key_btn = QPushButton("« Previous Key", nav_frame)
        self.prev_key_btn.clicked.connect(self.goto_prev_key)
        
        self.next_key_btn = QPushButton("Next Key »", nav_frame)
        self.next_key_btn.clicked.connect(self.goto_next_key)
        
        buttons_layout.addWidget(self.prev_key_btn)
        buttons_layout.addWidget(self.next_key_btn)
        nav_layout.addLayout(buttons_layout)
        
        right_layout.addWidget(nav_frame)
        
        # Analysis results
        analysis_frame = QFrame(right_panel)
        analysis_frame.setFrameShape(QFrame.Shape.StyledPanel)
        analysis_layout = QVBoxLayout(analysis_frame)
        
        analysis_title = QLabel("Analysis Results", analysis_frame)
        analysis_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        analysis_layout.addWidget(analysis_title)
        
        self.analysis_text = QTextEdit(analysis_frame)
        self.analysis_text.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_text)
        
        right_layout.addWidget(analysis_frame)
        
        results_layout.addWidget(right_panel)
        
        layout.addWidget(self.results_frame)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.close_btn = ElegantButton("Close", self, self.dark_mode, is_primary=False)
        self.close_btn.clicked.connect(self.close)
        
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Apply theme
        self.update_theme()
        
    def toggle_max_frames(self):
        """Toggle between auto and manual max frames"""
        self.max_frames_spin.setEnabled(not self.auto_frames_check.isChecked())
        
    def update_theme(self):
        """Update UI theme"""
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        self.setStyleSheet(f"background-color: {theme['bg_primary']}; color: {theme['text_primary']};")
        
    def start_selection(self):
        """Start the frame selection process"""
        if not hasattr(self.parent_app, 'frames') or not self.parent_app.frames:
            QMessageBox.critical(self, "Error", "No frames loaded. Please load a TIF file first.")
            return
            
        # Hide settings frame, show progress frame
        self.settings_frame.setVisible(False)
        self.progress_frame.setVisible(True)
        
        # Initialize selector
        self.selector = IntelligentFrameSelector(self)
        self.selector.frames = self.parent_app.frames
        
        # Set parameters
        if self.auto_frames_check.isChecked():
            self.selector.max_frames = None
        else:
            self.selector.max_frames = self.max_frames_spin.value()
            
        self.selector.error_threshold = self.threshold_slider.value() / 100.0
        
        # Connect signals
        self.selector.progress_updated.connect(self.update_progress)
        self.selector.selection_complete.connect(self.on_selection_complete)
        self.selector.selection_error.connect(self.on_selection_error)
        
        # Start selection in a thread
        self.selection_thread = QThread()
        self.selector.moveToThread(self.selection_thread)
        self.selection_thread.started.connect(self.selector.select_frames)
        self.selection_thread.start()
        
    def update_progress(self, progress, status):
        """Update progress UI"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
        # Add to log
        import time
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {status}")
        
    def on_selection_complete(self, selected_frames):
        """Handle selection completion"""
        self.selection_thread.quit()
        self.selection_thread.wait()
        
        # Store results
        self.selected_frames = selected_frames
        
        # Hide progress frame, show results frame
        self.progress_frame.setVisible(False)
        self.results_frame.setVisible(True)
        
        # Update frames list
        self.frames_list.clear()
        for frame_idx in selected_frames:
            is_scene_change = frame_idx in self.selector.scene_changes
            score = self.selector.frame_scores.get(frame_idx, 0)
            item_text = f"Frame {frame_idx+1}"
            if is_scene_change:
                item_text += " [Scene Change]"
            item_text += f" (Score: {score:.2f})"
            
            self.frames_list.addItem(item_text)
            
        # Update analysis text
        analysis = (
            f"Scene Changes: {len(self.selector.scene_changes)} detected\n\n"
            f"Motion Complexity: {self._get_complexity_level()}\n\n"
            f"Frame Coverage: {len(selected_frames)} frames ({len(selected_frames)/len(self.parent_app.frames)*100:.1f}% of video)\n\n"
            f"Mean Error Threshold: {self.selector.error_threshold:.2f}\n\n"
        )
        
        self.analysis_text.setText(analysis)
        
        # Setup navigation
        self.frame_pos_slider.setRange(0, len(self.parent_app.frames) - 1)
        self.frame_pos_slider.setEnabled(True)
        self.frame_pos_slider.valueChanged.connect(self.on_slider_changed)
        
        # Enable buttons
        self.prev_key_btn.setEnabled(True)
        self.next_key_btn.setEnabled(True)
        
        # Select the first frame
        if selected_frames:
            self.frames_list.setCurrentRow(0)
            self.on_frame_selected(self.frames_list.item(0))
        
    def on_selection_error(self, error_msg):
        """Handle selection error"""
        self.selection_thread.quit()
        self.selection_thread.wait()
        
        # Show error
        QMessageBox.critical(self, "Error", f"Frame selection failed:\n{error_msg}")
        
        # Hide progress frame, show settings frame
        self.progress_frame.setVisible(False)
        self.settings_frame.setVisible(True)
        
    def _get_complexity_level(self):
        """Determine motion complexity level"""
        if not hasattr(self.selector, 'frame_scores') or not self.selector.frame_scores:
            return "Unknown"
            
        # Calculate average score
        avg_score = sum(self.selector.frame_scores.values()) / len(self.selector.frame_scores)
        
        if avg_score < 0.1:
            return "Low"
        elif avg_score < 0.2:
            return "Medium"
        else:
            return "High"
            
    def on_frame_selected(self, item):
        """Handle frame selection from list"""
        if not item:
            return
            
        # Extract frame index from item text
        text = item.text()
        frame_idx = int(text.split(" ")[1]) - 1  # Convert from 1-based to 0-based
        
        # Update parent app to show this frame
        if hasattr(self.parent_app, 'set_current_frame'):
            self.parent_app.set_current_frame(frame_idx)
            
        # Update slider
        self.frame_pos_slider.setValue(frame_idx)
        
        # Update position label
        total_frames = len(self.parent_app.frames) if hasattr(self.parent_app, 'frames') else 0
        self.frame_pos_label.setText(f"Current Position: Frame {frame_idx+1}/{total_frames}")
        
    def on_slider_changed(self, value):
        """Handle slider movement"""
        # Update parent app
        if hasattr(self.parent_app, 'set_current_frame'):
            self.parent_app.set_current_frame(value)
            
        # Update position label
        total_frames = len(self.parent_app.frames) if hasattr(self.parent_app, 'frames') else 0
        self.frame_pos_label.setText(f"Current Position: Frame {value+1}/{total_frames}")
        
        # Find and select corresponding item in list if it's a key frame
        for i in range(self.frames_list.count()):
            item = self.frames_list.item(i)
            text = item.text()
            frame_idx = int(text.split(" ")[1]) - 1
            
            if frame_idx == value:
                self.frames_list.setCurrentItem(item)
                break
                
    def goto_next_key(self):
        """Go to next key frame"""
        if not self.selected_frames:
            return
            
        # Get current frame
        current_frame = self.parent_app.current_frame_idx if hasattr(self.parent_app, 'current_frame_idx') else 0
        
        # Find next key frame
        next_frames = [f for f in self.selected_frames if f > current_frame]
        
        if next_frames:
            next_frame = min(next_frames)
            # Find and select corresponding item
            for i in range(self.frames_list.count()):
                item = self.frames_list.item(i)
                text = item.text()
                frame_idx = int(text.split(" ")[1]) - 1
                
                if frame_idx == next_frame:
                    self.frames_list.setCurrentItem(item)
                    self.on_frame_selected(item)
                    break
        
    def goto_prev_key(self):
        """Go to previous key frame"""
        if not self.selected_frames:
            return
            
        # Get current frame
        current_frame = self.parent_app.current_frame_idx if hasattr(self.parent_app, 'current_frame_idx') else 0
        
        # Find previous key frame
        prev_frames = [f for f in self.selected_frames if f < current_frame]
        
        if prev_frames:
            prev_frame = max(prev_frames)
            # Find and select corresponding item
            for i in range(self.frames_list.count()):
                item = self.frames_list.item(i)
                text = item.text()
                frame_idx = int(text.split(" ")[1]) - 1
                
                if frame_idx == prev_frame:
                    self.frames_list.setCurrentItem(item)
                    self.on_frame_selected(item)
                    break



