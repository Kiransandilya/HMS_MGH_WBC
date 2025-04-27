# complete_cell_analysis.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QSpinBox, QDoubleSpinBox, QProgressBar, 
    QGroupBox, QMessageBox, QTabWidget, QScrollArea, QComboBox,
    QSizePolicy, QFrame, QSplitter, QGridLayout, QApplication, QCheckBox
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QEvent,
    QPropertyAnimation, QParallelAnimationGroup
)
from PyQt6.QtGui import QImage, QPixmap, QColor, QPen, QFont
import pyqtgraph as pg

# Scientific libraries
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path
import threading
import tifffile as tiff
import os
import shutil
from skimage import io, feature
import time

# Optional dependencies
try:
    import imagej
    import scyjava
    IMAGEJ_AVAILABLE = True
except ImportError:
    print("Warning: ImageJ libraries not found. Install with 'pip install pyimagej'")
    IMAGEJ_AVAILABLE = False

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    print("Warning: deep_sort_realtime not found. Install with 'pip install deep-sort-realtime'")
    DEEPSORT_AVAILABLE = False

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except ImportError:
    print("Warning: QtWebEngine not found. Some visualizations will be disabled.")
    WEBENGINE_AVAILABLE = False

# Import from core module
from complete_cell_analysis_core import (
    CellTrackingPipeline, ImageViewer, CellMetricsViewer, BlobParameters, 
    TrackerParameters, ProcessingParameters, ProgressSignal
)

# Type hints
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any


class BlobData:
    """Class to store information about detected blobs"""
    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        frame_index: int,
        intensity_stats: dict
    ):
        self.x = x
        self.y = y
        self.radius = radius
        self.frame_index = frame_index
        self.area = np.pi * (radius ** 2)
        self.intensity_stats = intensity_stats
    
    def to_bbox(self) -> List[float]:
        """Convert blob data to bounding box format [x1, y1, width, height]"""
        return [
            self.x - self.radius,
            self.y - self.radius,
            2 * self.radius,
            2 * self.radius
        ]


class ImageProcessor:
    """Class for image processing operations"""
    def enhance_white_spots(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enhance white spots in the image"""
        # Apply Gaussian blur
        kernel_size = 3
        sigma = 0.5
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        
        # Simple thresholding
        _, white_mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        
        # Dilation
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(white_mask, kernel, iterations=1)
        
        return blurred, white_mask, dilated_mask


class BlobDetector:
    """Main class for blob detection and processing"""
    def __init__(self, blob_params: BlobParameters, visualize: bool = False):
        self.blob_params = blob_params
        self.visualize = visualize
        self.processor = ImageProcessor()

    def _calculate_intensity_stats(self, frame: np.ndarray, blob: np.ndarray) -> dict:
        """Calculate intensity statistics for a blob region"""
        y, x, r = blob
        y, x = int(y), int(x)
        y_min = max(0, y - int(r))
        y_max = min(frame.shape[0], y + int(r))
        x_min = max(0, x - int(r))
        x_max = min(frame.shape[1], x + int(r))
        
        blob_region = frame[y_min:y_max, x_min:x_max]
        blob_mask = np.zeros_like(blob_region, dtype=bool)
        
        # Handle potentially empty regions
        if blob_region.shape[0] == 0 or blob_region.shape[1] == 0:
            return {'mean': 0, 'max': 0, 'min': 0, 'std': 0}
            
        yy, xx = np.ogrid[:blob_region.shape[0], :blob_region.shape[1]]
        mask_center_y = blob_region.shape[0] // 2
        mask_center_x = blob_region.shape[1] // 2
        
        blob_mask[(yy - mask_center_y) ** 2 + (xx - mask_center_x) ** 2 <= r ** 2] = True
        
        blob_pixels = blob_region[blob_mask]
        
        if blob_pixels.size > 0:
            return {
                'mean': np.mean(blob_pixels),
                'max': np.max(blob_pixels),
                'min': np.min(blob_pixels),
                'std': np.std(blob_pixels)
            }
        return {'mean': 0, 'max': 0, 'min': 0, 'std': 0}

    def detect_blobs(self, mask: np.ndarray, frame_index: int, frame: np.ndarray) -> List[BlobData]:
        """Detect blobs in a single frame"""
        blobs_log = feature.blob_log(
            mask,
            min_sigma=self.blob_params.min_sigma,
            max_sigma=self.blob_params.max_sigma,
            num_sigma=self.blob_params.num_sigma,
            threshold=self.blob_params.threshold
        )
        
        blob_data_list = []
        for blob in blobs_log:
            intensity_stats = self._calculate_intensity_stats(frame, blob)
            blob_data = BlobData(
                x=blob[1],
                y=blob[0],
                radius=blob[2],
                frame_index=frame_index,
                intensity_stats=intensity_stats
            )
            blob_data_list.append(blob_data)
        
        return blob_data_list

    def create_overlay(self, frame: np.ndarray, blob_data_list: List[BlobData]) -> np.ndarray:
        """Create an overlay of detected blobs on the frame"""
        overlay = np.copy(frame)
        for blob in blob_data_list:
            cv2.circle(
                overlay,
                (int(blob.x), int(blob.y)),
                int(blob.radius * 1.3),
                (255),
                thickness=2
            )
        return overlay


# Helper class for collapsible panels
class CollapsibleBox(QFrame):
    """Custom collapsible box widget"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        
        self.toggle_button = QPushButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setText(f"{title} ▼")
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                border: none;
                background: transparent;
                color: white;
                font-weight: bold;
            }
        """)
        
        self.toggle_animation = QParallelAnimationGroup(self)
        
        self.content_area = QScrollArea()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.content_area.setFrameShape(QFrame.Shape.NoFrame)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)
        
        # Create animations and add them to the group
        self.min_height_animation = QPropertyAnimation(self, b"minimumHeight")
        self.max_height_animation = QPropertyAnimation(self, b"maximumHeight")
        self.content_animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        
        self.toggle_animation.addAnimation(self.min_height_animation)
        self.toggle_animation.addAnimation(self.max_height_animation)
        self.toggle_animation.addAnimation(self.content_animation)
        
        self.toggle_button.clicked.connect(self.on_toggle)
    
    def on_toggle(self, checked):
        """Toggle content visibility"""
        arrow = "▼" if checked else "►"
        title = self.toggle_button.text().split(" ")[0:-1]
        self.toggle_button.setText(f"{' '.join(title)} {arrow}")
        
        content_height = self.content_area.sizeHint().height()
        
        # Configure each animation individually
        # Content area animation
        self.content_animation.setDuration(150)
        self.content_animation.setStartValue(0 if checked else content_height)
        self.content_animation.setEndValue(content_height if checked else 0)
        
        # Widget minimum height animation
        self.min_height_animation.setDuration(150)
        self.min_height_animation.setStartValue(self.minimumSizeHint().height())
        self.min_height_animation.setEndValue(self.sizeHint().height() if checked else self.minimumSizeHint().height())
        
        # Widget maximum height animation
        self.max_height_animation.setDuration(150)
        self.max_height_animation.setStartValue(self.minimumSizeHint().height())
        self.max_height_animation.setEndValue(self.sizeHint().height() if checked else self.minimumSizeHint().height())
        
        self.toggle_animation.start()
    
    def setContentLayout(self, layout):
        """Set the layout of the content area"""
        self.content_area.setLayout(layout)
        
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = layout.sizeHint().height()
        
        # Configure animations individually
        self.content_animation.setStartValue(0)
        self.content_animation.setEndValue(content_height)
        
        self.min_height_animation.setStartValue(collapsed_height)
        self.min_height_animation.setEndValue(collapsed_height + content_height)
        
        self.max_height_animation.setStartValue(collapsed_height)
        self.max_height_animation.setEndValue(collapsed_height + content_height)
        
        self.toggle_animation.start()
        self.toggle_animation.setCurrentTime(self.toggle_animation.totalDuration())


class CompleteCellAnalysisScreen(QWidget):
    """Screen for complete cell analysis"""
    
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.parent_app = parent  # Store reference to parent app
        self.init_ui()
    
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Theme toggle in top bar
        from main import ThemeToggle, StyleSheet, ElegantButton
        
        # Create header bar
        header_bar = QFrame()
        header_bar.setStyleSheet(f"background-color: #1d1d1f;")
        header_bar.setFixedHeight(50)
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        # Title
        self.title_label = QLabel("Complete Cell Analysis", self)
        font = self.title_label.font()
        font.setPointSize(18)
        font.setWeight(font.Weight.Medium)
        self.title_label.setFont(font)
        header_layout.addWidget(self.title_label)
        
        # Back button
        self.back_button = ElegantButton("Back", self, self.dark_mode, is_primary=False, icon="back")
        self.back_button.setFixedSize(80, 36)
        self.back_button.clicked.connect(self.on_back_clicked)
        header_layout.addWidget(self.back_button)
        
        # Add tabs selector in header
        self.tabs_selector = QFrame()
        self.tabs_selector.setStyleSheet("background-color: #2d2d30; border-radius: 5px;")
        tabs_layout = QHBoxLayout(self.tabs_selector)
        tabs_layout.setContentsMargins(1, 1, 1, 1)
        tabs_layout.setSpacing(0)
        
        # Tab buttons
        self.tab_tracking_btn = QPushButton("Cell Tracking")
        self.tab_metrics_btn = QPushButton("Cell Metrics")
        self.tab_analysis_btn = QPushButton("Analysis")
        
        for btn in [self.tab_tracking_btn, self.tab_metrics_btn, self.tab_analysis_btn]:
            btn.setFixedHeight(32)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #bbbbbb;
                    border: none;
                    padding: 4px 12px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    color: white;
                }
                QPushButton:checked {
                    background-color: #0071e3;
                    color: white;
                    border-radius: 4px;
                }
            """)
            btn.setCheckable(True)
            tabs_layout.addWidget(btn)
        
        self.tab_tracking_btn.setChecked(True)
        
        # Connect tab buttons
        self.tab_tracking_btn.clicked.connect(lambda: self.switch_tab(0))
        self.tab_metrics_btn.clicked.connect(lambda: self.switch_tab(1))
        self.tab_analysis_btn.clicked.connect(lambda: self.switch_tab(2))
        
        header_layout.addWidget(self.tabs_selector)
        header_layout.addStretch()
        
        # Status label
        self.status_label = QLabel("Ready")
        header_layout.addWidget(self.status_label)
        
        # Add theme toggle
        self.theme_toggle = ThemeToggle(self, self.dark_mode)
        header_layout.addWidget(self.theme_toggle)
        
        # Add header to main layout
        main_layout.addWidget(header_bar)
        
        # Setup main content area
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabBarAutoHide(True)  # Hide the tab bar, we'll use our custom selector
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
            }
        """)

        # Create metrics viewer before setup_analysis_ui
        self.metrics_viewer = CellMetricsViewer()
    
        # Create tracking tab metrics viewer as well
        self.tracking_tab_metrics = CellMetricsViewer()
    
        
        # Setup tabs content
        self.setup_analysis_ui()
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)

        self.viewer.frame_changed.connect(self.on_frame_changed)
        self.viewer.frame_slider.valueChanged.connect(self.metrics_viewer.on_slider_changed)
        self.metrics_cell_combo.currentIndexChanged.connect(self.on_metrics_cell_selected)
        
        # Initialize pipeline
        self.progress_signal = ProgressSignal()
        self.progress_signal.progress.connect(self.update_progress)
        self.pipeline = CellTrackingPipeline(self.progress_signal)

        # Add this to fix the path issue:
        self.pipeline.temp_dir = Path(os.path.abspath("temp"))
        self.pipeline.temp_dir.mkdir(exist_ok=True)
        
        # Apply theme
        self.update_theme()
    
    def setup_analysis_ui(self):
        """Set up the cell analysis UI components for all tabs"""
        
        # First tab - Cell Tracking
        tracking_tab = QWidget()
        tracking_layout = QVBoxLayout(tracking_tab)
        tracking_layout.setContentsMargins(0, 0, 0, 0)
        tracking_layout.setSpacing(0)
        
        # Create main horizontal layout with sidebar and content
        main_horizontal_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Sidebar for controls
        self.sidebar = QFrame()
        self.sidebar.setMinimumWidth(250)
        self.sidebar.setMaximumWidth(350)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        # Sidebar collapse button
        sidebar_header = QFrame()
        sidebar_header_layout = QHBoxLayout(sidebar_header)
        sidebar_header_layout.setContentsMargins(0, 0, 0, 0)
        
        sidebar_header_layout.addStretch()
        self.sidebar_toggle_btn = QPushButton("←")
        self.sidebar_toggle_btn.setFixedSize(24, 24)
        self.sidebar_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d30;
                color: white;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
        """)
        self.sidebar_toggle_btn.clicked.connect(self.toggle_sidebar)
        sidebar_header_layout.addWidget(self.sidebar_toggle_btn)
        
        sidebar_layout.addWidget(sidebar_header)
        
        # Add control sections to sidebar
        self.add_sidebar_controls(sidebar_layout)
        
        # Content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)
        
        # Create vertical splitter for visualization and charts
        main_vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Cell visualization area (40% height)
        self.visualization_widget = QFrame()
        self.visualization_widget.setStyleSheet("background-color: black;")
        viz_layout = QVBoxLayout(self.visualization_widget)
        
        # Add image viewer
        self.viewer = ImageViewer()
        viz_layout.addWidget(self.viewer)
        
        # Charts section (60% height)
        charts_widget = QWidget()
        charts_layout = QGridLayout(charts_widget)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(6)
        
        # Create resizable chart panels
        self.position_chart = self.create_resizable_chart("Position")
        self.path_chart = self.create_resizable_chart("Path")
        self.velocity_chart = self.create_resizable_chart("Velocity")
        self.acceleration_chart = self.create_resizable_chart("Acceleration")
        
        charts_layout.addWidget(self.position_chart, 0, 0)
        charts_layout.addWidget(self.path_chart, 0, 1)
        charts_layout.addWidget(self.velocity_chart, 1, 0)
        charts_layout.addWidget(self.acceleration_chart, 1, 1)
        
        # Add widgets to vertical splitter
        main_vertical_splitter.addWidget(self.visualization_widget)
        main_vertical_splitter.addWidget(charts_widget)
        
        # Set initial size ratio (40% visualization, 60% charts)
        main_vertical_splitter.setSizes([40, 60])
        
        # Add vertical splitter to content layout
        content_layout.addWidget(main_vertical_splitter)
        
        # Add stats panel at bottom
        self.stats_panel = self.create_collapsible_stats_panel()
        content_layout.addWidget(self.stats_panel)
        
        # Add sidebar and content to horizontal splitter
        main_horizontal_splitter.addWidget(self.sidebar)
        main_horizontal_splitter.addWidget(content_widget)
        
        # Set initial size ratio for horizontal splitter (1:4)
        main_horizontal_splitter.setSizes([1, 4])
        
        # Add horizontal splitter to tracking layout
        tracking_layout.addWidget(main_horizontal_splitter)
        
        # Add tracking tab to tab widget
        self.tab_widget.addTab(tracking_tab, "Cell Tracking")
        
        # Setup other tabs
        self.setup_metrics_tab()
        self.setup_analysis_tab()
        
        # Connect signals
        # Connect signals AFTER all components are created
        self.viewer.frame_changed.connect(self.on_frame_changed)
        self.viewer.frame_slider.valueChanged.connect(self.metrics_viewer.on_slider_changed)
        self.metrics_cell_combo.currentIndexChanged.connect(self.on_metrics_cell_selected)
    
        # Connect cell selection between components
        self.cell_combo.currentIndexChanged.connect(self.on_cell_selected)

    def create_collapsible_group(self, title):
        """Create a collapsible group for parameters"""
        group = CollapsibleBox(title)
        group.setStyleSheet("""
            QFrame {
                background-color: #2d2d30;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
            }
            QLabel {
                color: #bbbbbb;
            }
            QSpinBox, QDoubleSpinBox, QCheckBox {
                background-color: #1d1d1f;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 3px;
            }
        """)
        return group

    def setup_metrics_tab(self):
        """Set up the cell metrics tab"""
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create scroll content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(20, 20, 20, 20)
        scroll_layout.setSpacing(20)
        
        # Cell selector and controls
        controls_layout = QHBoxLayout()
        
        cell_label = QLabel("Select Cell:")
        self.metrics_cell_combo = QComboBox()
        self.metrics_cell_combo.setFixedWidth(150)
        self.metrics_cell_combo.currentIndexChanged.connect(self.on_cell_selected)
        
        controls_layout.addWidget(cell_label)
        controls_layout.addWidget(self.metrics_cell_combo)
        controls_layout.addStretch()
        
        export_btn = QPushButton("Export Data")
        export_btn.setFixedSize(120, 32)
        report_btn = QPushButton("Generate Report")
        report_btn.setFixedSize(120, 32)
        
        controls_layout.addWidget(export_btn)
        controls_layout.addWidget(report_btn)
        
        scroll_layout.addLayout(controls_layout)
        
        # Large metrics charts
        charts_grid = QGridLayout()
        charts_grid.setSpacing(15)
        
        # Position chart
        position_container = QFrame()
        position_container.setStyleSheet("background-color: #2d2d30; border-radius: 6px;")
        position_container.setMinimumHeight(250)
        position_layout = QVBoxLayout(position_container)
        
        position_title = QLabel("Position Tracking")
        position_title.setStyleSheet("color: white; font-weight: bold;")
        position_layout.addWidget(position_title)
        
        self.metrics_position_chart = pg.PlotWidget()
        self.setup_metrics_position_chart(self.metrics_position_chart)
        position_layout.addWidget(self.metrics_position_chart)
        
        # Path chart
        path_container = QFrame()
        path_container.setStyleSheet("background-color: #2d2d30; border-radius: 6px;")
        path_container.setMinimumHeight(250)
        path_layout = QVBoxLayout(path_container)
        
        path_title = QLabel("Movement Path")
        path_title.setStyleSheet("color: white; font-weight: bold;")
        path_layout.addWidget(path_title)
        
        self.metrics_path_chart = pg.PlotWidget()
        self.setup_metrics_path_chart(self.metrics_path_chart)
        path_layout.addWidget(self.metrics_path_chart)
        
        # Velocity chart
        velocity_container = QFrame()
        velocity_container.setStyleSheet("background-color: #2d2d30; border-radius: 6px;")
        velocity_container.setMinimumHeight(250)
        velocity_layout = QVBoxLayout(velocity_container)
        
        velocity_title = QLabel("Velocity Profile")
        velocity_title.setStyleSheet("color: white; font-weight: bold;")
        velocity_layout.addWidget(velocity_title)
        
        self.metrics_velocity_chart = pg.PlotWidget()
        self.setup_metrics_velocity_chart(self.metrics_velocity_chart)
        velocity_layout.addWidget(self.metrics_velocity_chart)
        
        # Acceleration chart
        acceleration_container = QFrame()
        acceleration_container.setStyleSheet("background-color: #2d2d30; border-radius: 6px;")
        acceleration_container.setMinimumHeight(250)
        acceleration_layout = QVBoxLayout(acceleration_container)
        
        acceleration_title = QLabel("Acceleration Profile")
        acceleration_title.setStyleSheet("color: white; font-weight: bold;")
        acceleration_layout.addWidget(acceleration_title)
        
        self.metrics_acceleration_chart = pg.PlotWidget()
        self.setup_metrics_acceleration_chart(self.metrics_acceleration_chart)
        acceleration_layout.addWidget(self.metrics_acceleration_chart)
        
        # Add charts to grid
        charts_grid.addWidget(position_container, 0, 0)
        charts_grid.addWidget(path_container, 0, 1)
        charts_grid.addWidget(velocity_container, 1, 0)
        charts_grid.addWidget(acceleration_container, 1, 1)
        
        scroll_layout.addLayout(charts_grid)
        
        # Statistics summary section
        stats_summary = QFrame()
        stats_summary.setStyleSheet("background-color: #2d2d30; border-radius: 6px;")
        stats_layout = QVBoxLayout(stats_summary)
        
        summary_title = QLabel("Cell Statistics Summary")
        summary_title.setStyleSheet("color: white; font-weight: bold;")
        stats_layout.addWidget(summary_title)
        
        # Statistics in 3 columns
        stats_grid = QGridLayout()
        stats_grid.setColumnStretch(0, 1)
        stats_grid.setColumnStretch(1, 1)
        stats_grid.setColumnStretch(2, 1)
        
        # Add statistics data (will be populated dynamically)
        self.add_metric_stat(stats_grid, 0, 0, "Frames Tracked:", "-")
        self.add_metric_stat(stats_grid, 1, 0, "Track Stability:", "-")
        self.add_metric_stat(stats_grid, 0, 1, "Average Velocity:", "-")
        self.add_metric_stat(stats_grid, 1, 1, "Max Velocity:", "-")
        self.add_metric_stat(stats_grid, 0, 2, "Total Distance:", "-")
        self.add_metric_stat(stats_grid, 1, 2, "Net Displacement:", "-")
        
        stats_layout.addLayout(stats_grid)
        scroll_layout.addWidget(stats_summary)
        
        # Set scroll content and add to tab
        scroll_area.setWidget(scroll_content)
        metrics_layout.addWidget(scroll_area)
        
        self.tab_widget.addTab(metrics_tab, "Cell Metrics")

    def setup_analysis_tab(self):
        """Set up the analysis tab UI"""
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        
        # Create a scroll area for the tab content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create scroll content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(20, 20, 20, 20)
        scroll_layout.setSpacing(20)
        
        # Add title
        title_label = QLabel("Advanced Cell Analysis")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setWeight(QFont.Weight.Medium)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: white;")
        scroll_layout.addWidget(title_label)
        
        # Add analysis options
        options_container = QFrame()
        options_container.setStyleSheet("background-color: #2d2d30; border-radius: 6px;")
        options_layout = QVBoxLayout(options_container)
        
        options_title = QLabel("Analysis Options")
        options_title.setStyleSheet("color: white; font-weight: bold;")
        options_layout.addWidget(options_title)
        
        # Analysis parameters grid
        params_grid = QGridLayout()
        params_grid.setColumnStretch(0, 1)
        params_grid.setColumnStretch(1, 1)
        
        # Add batch analysis option
        batch_checkbox = QCheckBox("Enable Batch Analysis")
        batch_checkbox.setStyleSheet("color: white;")
        params_grid.addWidget(batch_checkbox, 0, 0)
        
        # Add time series analysis option
        timeseries_checkbox = QCheckBox("Enable Time Series Analysis")
        timeseries_checkbox.setStyleSheet("color: white;")
        params_grid.addWidget(timeseries_checkbox, 0, 1)
        
        # Add statistical analysis option
        stats_checkbox = QCheckBox("Perform Statistical Analysis")
        stats_checkbox.setStyleSheet("color: white;")
        params_grid.addWidget(stats_checkbox, 1, 0)
        
        # Add export option
        export_checkbox = QCheckBox("Export Results")
        export_checkbox.setStyleSheet("color: white;")
        params_grid.addWidget(export_checkbox, 1, 1)
        
        options_layout.addLayout(params_grid)
        scroll_layout.addWidget(options_container)
        
        # Add comparative analysis section
        comparison_container = QFrame()
        comparison_container.setStyleSheet("background-color: #2d2d30; border-radius: 6px;")
        comparison_container.setMinimumHeight(300)
        comparison_layout = QVBoxLayout(comparison_container)
        
        comparison_title = QLabel("Comparative Analysis")
        comparison_title.setStyleSheet("color: white; font-weight: bold;")
        comparison_layout.addWidget(comparison_title)
        
        # Add cell selection for comparison
        comparison_controls = QHBoxLayout()
        
        comparison_label = QLabel("Compare Cells:")
        comparison_label.setStyleSheet("color: white;")
        comparison_controls.addWidget(comparison_label)
        
        cell1_combo = QComboBox()
        cell1_combo.setFixedWidth(100)
        comparison_controls.addWidget(cell1_combo)
        
        comparison_controls.addWidget(QLabel("vs"))
        
        cell2_combo = QComboBox()
        cell2_combo.setFixedWidth(100)
        comparison_controls.addWidget(cell2_combo)
        
        comparison_btn = QPushButton("Compare")
        comparison_btn.setFixedWidth(100)
        comparison_controls.addWidget(comparison_btn)
        
        comparison_controls.addStretch()
        
        comparison_layout.addLayout(comparison_controls)
        
        # Create comparison chart
        comparison_chart = pg.PlotWidget()
        comparison_chart.setBackground('k')
        comparison_chart.setMinimumHeight(200)
        comparison_chart.showGrid(x=True, y=True, alpha=0.3)
        comparison_layout.addWidget(comparison_chart)
        
        scroll_layout.addWidget(comparison_container)
        
        # Add reporting section
        reporting_container = QFrame()
        reporting_container.setStyleSheet("background-color: #2d2d30; border-radius: 6px;")
        reporting_layout = QVBoxLayout(reporting_container)
        
        reporting_title = QLabel("Report Generation")
        reporting_title.setStyleSheet("color: white; font-weight: bold;")
        reporting_layout.addWidget(reporting_title)
        
        # Report options
        report_options_layout = QHBoxLayout()
        
        report_type_label = QLabel("Report Type:")
        report_type_label.setStyleSheet("color: white;")
        report_options_layout.addWidget(report_type_label)
        
        report_type_combo = QComboBox()
        report_type_combo.addItems(["Summary Report", "Detailed Report", "Technical Report", "Publication Ready"])
        report_options_layout.addWidget(report_type_combo)
        
        report_options_layout.addStretch()
        
        generate_btn = QPushButton("Generate Report")
        generate_btn.setFixedWidth(150)
        report_options_layout.addWidget(generate_btn)
        
        reporting_layout.addLayout(report_options_layout)
        
        scroll_layout.addWidget(reporting_container)
        
        # Add run analysis button
        run_btn = QPushButton("Run Advanced Analysis")
        run_btn.setMinimumHeight(40)
        run_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                          stop:0 #0071e3, stop:1 #42a1ec);
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                          stop:0 #1082f4, stop:1 #53b2fd);
            }
        """)
        scroll_layout.addWidget(run_btn)
        
        # Set scroll content and add to tab
        scroll_area.setWidget(scroll_content)
        analysis_layout.addWidget(scroll_area)
        
        self.tab_widget.addTab(analysis_tab, "Analysis")

    def switch_tab(self, index):
        """Switch active tab and update content"""
        self.tab_widget.setCurrentIndex(index)
        
        # Update tab button states
        buttons = [self.tab_tracking_btn, self.tab_metrics_btn, self.tab_analysis_btn]
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)

    def add_metric_stat(self, grid, row, col, label_text, value_text):
        """Add a statistic to the metrics summary grid"""
        stat_layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet("color: #bbbbbb;")
        value = QLabel(value_text)
        value.setStyleSheet("color: white;")
        stat_layout.addWidget(label)
        stat_layout.addWidget(value)
        stat_layout.addStretch()
        grid.addLayout(stat_layout, row, col)

    def setup_metrics_position_chart(self, chart):
        """Set up the detailed position chart for metrics tab"""
        chart.setBackground('k')
        chart.showGrid(x=True, y=True, alpha=0.3)
        chart.setLabel('left', 'Position (pixels)')
        chart.setLabel('bottom', 'Frame')
        
        # Style the chart
        chart.getAxis('bottom').setPen((128, 128, 128))
        chart.getAxis('left').setPen((128, 128, 128))
        chart.getAxis('bottom').setTextPen((200, 200, 200))
        chart.getAxis('left').setTextPen((200, 200, 200))
        
        # Create plot lines
        x_position_pen = pg.mkPen(color=(66, 153, 225), width=2)
        y_position_pen = pg.mkPen(color=(245, 101, 101), width=2)
        self.metrics_x_position_line = chart.plot([], [], pen=x_position_pen, name='X Position')
        self.metrics_y_position_line = chart.plot([], [], pen=y_position_pen, name='Y Position')
        
        # Add legend
        chart.addLegend()

    def setup_metrics_path_chart(self, chart):
        """Set up the detailed path chart for metrics tab"""
        chart.setBackground('k')
        chart.showGrid(x=True, y=True, alpha=0.3)
        chart.setLabel('left', 'Y Position (pixels)')
        chart.setLabel('bottom', 'X Position (pixels)')
        
        # Style the chart
        chart.getAxis('bottom').setPen((128, 128, 128))
        chart.getAxis('left').setPen((128, 128, 128))
        chart.getAxis('bottom').setTextPen((200, 200, 200))
        chart.getAxis('left').setTextPen((200, 200, 200))
        
        # Create plot for cell path
        path_pen = pg.mkPen(color=(104, 211, 145), width=2)
        self.metrics_path_line = chart.plot([], [], pen=path_pen, name='Path')
        
        # Create scatter plot for current position
        current_position_brush = pg.mkBrush(252, 129, 129)
        self.metrics_position_scatter = pg.ScatterPlotItem(size=10, brush=current_position_brush)
        chart.addItem(self.metrics_position_scatter)
        
        # Add legend
        chart.addLegend()

    def setup_metrics_velocity_chart(self, chart):
        """Set up the detailed velocity chart for metrics tab"""
        chart.setBackground('k')
        chart.showGrid(x=True, y=True, alpha=0.3)
        chart.setLabel('left', 'Velocity (px/frame)')
        chart.setLabel('bottom', 'Frame')
        
        # Style the chart
        chart.getAxis('bottom').setPen((128, 128, 128))
        chart.getAxis('left').setPen((128, 128, 128))
        chart.getAxis('bottom').setTextPen((200, 200, 200))
        chart.getAxis('left').setTextPen((200, 200, 200))
        
        # Create velocity line
        velocity_pen = pg.mkPen(color=(236, 201, 75), width=2)
        self.metrics_velocity_line = chart.plot([], [], pen=velocity_pen, name='Velocity')
        
        # Add average line
        avg_pen = pg.mkPen(color=(128, 90, 213), width=1.5, style=Qt.PenStyle.DashLine)
        self.metrics_velocity_avg_line = pg.InfiniteLine(angle=0, movable=False, pen=avg_pen)
        chart.addItem(self.metrics_velocity_avg_line)
        
        # Add legend
        chart.addLegend()

    def setup_metrics_acceleration_chart(self, chart):
        """Set up the detailed acceleration chart for metrics tab"""
        chart.setBackground('k')
        chart.showGrid(x=True, y=True, alpha=0.3)
        chart.setLabel('left', 'Acceleration (px/frame²)')
        chart.setLabel('bottom', 'Frame')
        
        # Style the chart
        chart.getAxis('bottom').setPen((128, 128, 128))
        chart.getAxis('left').setPen((128, 128, 128))
        chart.getAxis('bottom').setTextPen((200, 200, 200))
        chart.getAxis('left').setTextPen((200, 200, 200))
        
        # Add zero line
        zero_pen = pg.mkPen(color=(128, 128, 128), width=1, style=Qt.PenStyle.DashLine)
        chart.addLine(y=0, pen=zero_pen)
        
        # Create acceleration line
        acceleration_pen = pg.mkPen(color=(213, 63, 140), width=2)
        self.metrics_acceleration_line = chart.plot([], [], pen=acceleration_pen, name='Acceleration')
        
        # Add legend
        chart.addLegend()

    def update_theme(self):
        """Update UI elements with current theme"""
        try:
            from main import StyleSheet
            theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
            
            # Update header bar
            header_bar_style = f"background-color: {theme['title_bar']};"
            self.title_label.setStyleSheet(f"color: {theme['text_primary']};")
            
            # Update tab buttons
            tabs_selector_style = f"background-color: {theme['bg_secondary']}; border-radius: 5px;"
            self.tabs_selector.setStyleSheet(tabs_selector_style)
            
            # Update sidebar
            sidebar_style = f"background-color: {theme['bg_secondary']};"
            self.sidebar.setStyleSheet(sidebar_style)
            
            # Update visualization widget
            self.visualization_widget.setStyleSheet("background-color: black;")
            
            # Update buttons with new theme
            if hasattr(self, 'back_button'):
                self.back_button.dark_mode = self.dark_mode
                self.back_button.update_style()
            
            if hasattr(self, 'process_button'):
                self.process_button.setStyleSheet(f"background-color: {theme['accent']}; color: white;")
            
            if hasattr(self, 'save_button'):
                self.save_button.setStyleSheet(f"background-color: {theme['bg_secondary']}; color: {theme['text_primary']};")
            
            # Update charts with theme
            for plot in [self.position_chart, self.path_chart, self.velocity_chart, self.acceleration_chart]:
                if hasattr(plot, 'plot_widget'):
                    plot.plot_widget.setBackground('k' if self.dark_mode else 'w')
        except Exception as e:
            print(f"Error updating theme: {str(e)}")

    def create_resizable_chart(self, title):
        """Create a resizable chart panel with controls"""
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: #111111;
                border: 1px solid #333333;
                border-radius: 4px;
            }
        """)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with title and resize controls
        header = QWidget()
        header.setStyleSheet("background-color: #1a1a1a; border-bottom: 1px solid #333333;")
        header.setFixedHeight(30)
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        
        resize_layout = QHBoxLayout()
        resize_layout.setSpacing(2)
        
        decrease_btn = QPushButton("-")
        decrease_btn.setFixedSize(20, 20)
        decrease_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: white;
                border-radius: 2px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        
        increase_btn = QPushButton("+")
        increase_btn.setFixedSize(20, 20)
        increase_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: white;
                border-radius: 2px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        
        resize_layout.addWidget(decrease_btn)
        resize_layout.addWidget(increase_btn)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addLayout(resize_layout)
        
        layout.addWidget(header)
        
        # Chart content
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('k')
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Style the chart
        plot_widget.getAxis('bottom').setPen((128, 128, 128))
        plot_widget.getAxis('left').setPen((128, 128, 128))
        plot_widget.getAxis('bottom').setTextPen((200, 200, 200))
        plot_widget.getAxis('left').setTextPen((200, 200, 200))
        
        # Set minimum height
        plot_widget.setMinimumHeight(140)

        # Store references for resize functionality
        container.plot_widget = plot_widget
        container.decrease_btn = decrease_btn
        container.increase_btn = increase_btn
        
        
        # Configure chart based on type
        if title == "Position":
            self.setup_position_chart(plot_widget)
        elif title == "Path":
            self.setup_path_chart(plot_widget)
        elif title == "Velocity":
            self.setup_velocity_chart(plot_widget)
        elif title == "Acceleration":
            self.setup_acceleration_chart(plot_widget)
        
        layout.addWidget(plot_widget)
        
 
        # Connect resize buttons
        decrease_btn.clicked.connect(lambda: self.resize_chart(container, -20))
        increase_btn.clicked.connect(lambda: self.resize_chart(container, 20))
        
        return container

    def create_collapsible_stats_panel(self):
        """Create the collapsible statistics panel"""
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: #111111;
                border: 1px solid #333333;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
        """)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with title and toggle button
        header = QWidget()
        header.setStyleSheet("background-color: #1a1a1a; border-bottom: 1px solid #333333;")
        header.setFixedHeight(30)
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        
        title_label = QLabel("Real-time Cell Statistics")
        title_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        
        # Cell selector
        cell_selector_layout = QHBoxLayout()
        cell_selector_layout.setContentsMargins(0, 0, 0, 0)
        cell_selector_layout.setSpacing(6)
        
        cell_label = QLabel("Select Cell:")
        cell_label.setStyleSheet("color: #bbbbbb; font-size: 12px;")
        
        self.cell_combo = QComboBox()
        self.cell_combo.setFixedWidth(100)
        self.cell_combo.setStyleSheet("""
            QComboBox {
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 1px 6px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555555;
            }
        """)
        
        cell_selector_layout.addWidget(cell_label)
        cell_selector_layout.addWidget(self.cell_combo)
        
        self.toggle_button = QPushButton("▼")
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
            }
        """)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addLayout(cell_selector_layout)
        header_layout.addWidget(self.toggle_button)
        
        layout.addWidget(header)
        
        # Statistics content
        self.stats_content = QWidget()
        stats_layout = QGridLayout(self.stats_content)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        stats_layout.setSpacing(8)
        
        # Add statistics labels - 3 columns
        self.stats_labels = {}
        
        # Column 1
        col1_layout = QVBoxLayout()
        stats_current_frame = self.create_stat_label("Current Frame:", "-")
        stats_position = self.create_stat_label("Position (x, y):", "-, -")
        col1_layout.addLayout(stats_current_frame["layout"])
        col1_layout.addLayout(stats_position["layout"])
        stats_layout.addLayout(col1_layout, 0, 0)
        
        # Column 2
        col2_layout = QVBoxLayout()
        stats_velocity = self.create_stat_label("Current Velocity:", "-")
        stats_acceleration = self.create_stat_label("Current Acceleration:", "-")
        col2_layout.addLayout(stats_velocity["layout"])
        col2_layout.addLayout(stats_acceleration["layout"])
        stats_layout.addLayout(col2_layout, 0, 1)
        
        # Column 3
        col3_layout = QVBoxLayout()
        stats_area = self.create_stat_label("Current Area:", "-")
        stats_total_distance = self.create_stat_label("Total Distance:", "-")
        stats_direction = self.create_stat_label("Movement Direction:", "-")
        col3_layout.addLayout(stats_area["layout"])
        col3_layout.addLayout(stats_total_distance["layout"]) 
        col3_layout.addLayout(stats_direction["layout"])
        stats_layout.addLayout(col3_layout, 0, 2)
        
        # Store the stat labels for updating later
        self.stats_labels['current_frame'] = stats_current_frame
        self.stats_labels['position'] = stats_position
        self.stats_labels['velocity'] = stats_velocity
        self.stats_labels['acceleration'] = stats_acceleration
        self.stats_labels['area'] = stats_area
        self.stats_labels['total_distance'] = stats_total_distance
        self.stats_labels['direction'] = stats_direction
        
        layout.addWidget(self.stats_content)
        
        # Connect toggle button
        self.toggle_button.clicked.connect(self.toggle_stats_panel)
        
        return container

    def create_stat_label(self, label_text, value_text):
        """Create a label pair for statistics"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        label = QLabel(label_text)
        label.setStyleSheet("color: #bbbbbb;")
        
        value = QLabel(value_text)
        value.setStyleSheet("color: #ffffff;")
        
        layout.addWidget(label)
        layout.addWidget(value, 1)
        
        return {"layout": layout, "label": label, "value": value}

    def toggle_stats_panel(self):
        """Toggle statistics panel visibility"""
        if self.stats_content.isVisible():
            self.stats_content.hide()
            self.toggle_button.setText("▲")
        else:
            self.stats_content.show()
            self.toggle_button.setText("▼")

    def toggle_sidebar(self):
        """Toggle sidebar visibility"""
        if self.sidebar.width() > 50:  # If sidebar is expanded
            self.sidebar.setFixedWidth(30)
            self.sidebar_toggle_btn.setText("→")
            
            # Hide all sidebar content except the toggle button
            for i in range(1, self.sidebar.layout().count()):
                item = self.sidebar.layout().itemAt(i)
                if item.widget():
                    item.widget().hide()
        else:  # If sidebar is collapsed
            self.sidebar.setFixedWidth(250)
            self.sidebar_toggle_btn.setText("←")
            
            # Show all sidebar content
            for i in range(1, self.sidebar.layout().count()):
                item = self.sidebar.layout().itemAt(i)
                if item.widget():
                    item.widget().show()

    def resize_chart(self, chart_container, change):
        """Resize a chart by the specified amount"""
        current_height = chart_container.plot_widget.height()
        new_height = max(120, current_height + change)  # Minimum height of 120px
        chart_container.plot_widget.setFixedHeight(new_height)

    def setup_position_chart(self, chart_widget):
        """Set up the position chart with stock-like styling"""
        chart_widget.setLabel('left', 'Position (pixels)')
        chart_widget.setLabel('bottom', 'Frame')
        
        # Create two plot lines for X and Y positions
        x_position_pen = pg.mkPen(color=(66, 153, 225), width=2)
        y_position_pen = pg.mkPen(color=(245, 101, 101), width=2)
        
        # Create plot lines
        self.x_position_line = chart_widget.plot([], [], pen=x_position_pen, name='X Position')
        self.y_position_line = chart_widget.plot([], [], pen=y_position_pen, name='Y Position')
        
        # Add legend
        chart_widget.addLegend()

    def setup_path_chart(self, chart_widget):
        """Set up the path chart"""
        chart_widget.setLabel('left', 'Y Position (pixels)')
        chart_widget.setLabel('bottom', 'X Position (pixels)')
        
        # Create plot for cell path
        path_pen = pg.mkPen(color=(104, 211, 145), width=2)
        self.path_line = chart_widget.plot([], [], pen=path_pen)
        
        # Create scatter plot for current position
        current_position_brush = pg.mkBrush(252, 129, 129)
        self.current_position_scatter = pg.ScatterPlotItem(size=10, brush=current_position_brush)
        chart_widget.addItem(self.current_position_scatter)

        # Add legend
        chart_widget.addLegend()


    def setup_velocity_chart(self, chart_widget):
        """Set up the velocity chart"""
        chart_widget.setLabel('left', 'Velocity (px/frame)')
        chart_widget.setLabel('bottom', 'Frame')
        
        # Create velocity line
        velocity_pen = pg.mkPen(color=(236, 201, 75), width=2)
        self.velocity_line = chart_widget.plot([], [], pen=velocity_pen, name='Velocity')
        
        # Create filled area under the curve
        self.velocity_fill = pg.FillBetweenItem(
            pg.PlotCurveItem([], []),
            pg.PlotCurveItem([], []),
            brush=pg.mkBrush(236, 201, 75, 50)
        )
        chart_widget.addItem(self.velocity_fill)
        # Add legend
        chart_widget.addLegend()

    def setup_acceleration_chart(self, chart_widget):
        """Set up the acceleration chart"""
        chart_widget.setLabel('left', 'Acceleration (px/frame²)')
        chart_widget.setLabel('bottom', 'Frame')
        
        # Add zero line
        zero_pen = pg.mkPen(color=(128, 128, 128), width=1, style=Qt.PenStyle.DashLine)
        chart_widget.addLine(y=0, pen=zero_pen)
        
        # Create acceleration line
        acceleration_pen = pg.mkPen(color=(213, 63, 140), width=2)
        self.acceleration_line = chart_widget.plot([], [], pen=acceleration_pen, name='Acceleration')

        # Add legend
        chart_widget.addLegend()

    def on_cell_selected(self, index):
        """Synchronize cell selection between components"""
        if self.cell_combo.count() == 0 or index < 0:
            return
            
        try:
            # Get selected cell ID
            cell_id = int(self.cell_combo.itemText(index).split()[1])
            
            # Block signals to prevent recursive calls
            self.metrics_cell_combo.blockSignals(True)
            
            # Sync with metrics tab combo if different
            metrics_index = self.metrics_cell_combo.findText(f"Cell {cell_id}")
            if metrics_index >= 0 and metrics_index != self.metrics_cell_combo.currentIndex():
                self.metrics_cell_combo.setCurrentIndex(metrics_index)
            
            # Restore signals
            self.metrics_cell_combo.blockSignals(False)
            
            # Update display for the selected cell
            self.update_cell_metrics(cell_id)
        except Exception as e:
            print(f"Error in on_cell_selected: {str(e)}")



    def on_back_clicked(self):
        """Return to analysis type screen"""
        if self.parent_app and hasattr(self.parent_app, 'show_analysis_type_screen'):
            self.parent_app.show_analysis_type_screen()

    def add_sidebar_controls(self, layout):
        """Add control sections to sidebar"""
        # File selection
        self.file_button = QPushButton("Select TIFF File")
        self.file_button.setMinimumHeight(36)
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)
        layout.addSpacing(15)
        
        # Preprocessing toggles
        preprocess_group = self.create_collapsible_group("Preprocessing Options")
        preprocess_layout = QVBoxLayout()
        
        # Kalman filter toggle
        self.use_kalman = QCheckBox("Use Kalman Filtering")
        self.use_kalman.setChecked(False)
        self.use_kalman.stateChanged.connect(self.toggle_kalman_options)
        preprocess_layout.addWidget(self.use_kalman)
        
        # Kalman parameters (initially hidden)
        self.kalman_params_widget = QWidget()
        kalman_params_layout = QVBoxLayout(self.kalman_params_widget)
        kalman_params_layout.setContentsMargins(10, 0, 0, 0)
        
        # Acquisition noise
        kalman_params_layout.addWidget(QLabel("Acquisition Noise:"))
        self.acquisition_noise = QDoubleSpinBox()
        self.acquisition_noise.setRange(0.001, 1.0)
        self.acquisition_noise.setValue(0.05)
        self.acquisition_noise.setSingleStep(0.01)
        kalman_params_layout.addWidget(self.acquisition_noise)
        
        # Kalman bias
        kalman_params_layout.addWidget(QLabel("Kalman Bias:"))
        self.kalman_bias = QDoubleSpinBox()
        self.kalman_bias.setRange(0.1, 1.0)
        self.kalman_bias.setValue(0.8)
        self.kalman_bias.setSingleStep(0.1)
        kalman_params_layout.addWidget(self.kalman_bias)
        
        # Initially hide kalman parameters
        self.kalman_params_widget.setVisible(False)
        preprocess_layout.addWidget(self.kalman_params_widget)
        
        # ImageJ processing toggle
        self.use_imagej = QCheckBox("Use ImageJ Processing")
        self.use_imagej.setChecked(False)
        preprocess_layout.addWidget(self.use_imagej)
        
        preprocess_group.setContentLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        layout.addSpacing(10)
        
        # Blob Detection Parameters
        blob_group = self.create_collapsible_group("Blob Detection Parameters")
        blob_layout = QVBoxLayout()
        
        # Min Sigma
        blob_layout.addWidget(QLabel("Min Sigma:"))
        self.min_sigma = QDoubleSpinBox()
        self.min_sigma.setRange(1, 100)
        self.min_sigma.setValue(15)
        blob_layout.addWidget(self.min_sigma)
        
        # Max Sigma
        blob_layout.addWidget(QLabel("Max Sigma:"))
        self.max_sigma = QDoubleSpinBox()
        self.max_sigma.setRange(1, 100)
        self.max_sigma.setValue(30)
        blob_layout.addWidget(self.max_sigma)
        
        # Threshold
        blob_layout.addWidget(QLabel("Threshold:"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.001, 1.0)
        self.threshold.setValue(0.1)
        blob_layout.addWidget(self.threshold)
        
        # Num Sigma
        blob_layout.addWidget(QLabel("Num Sigma:"))
        self.num_sigma = QSpinBox()
        self.num_sigma.setRange(1, 20)
        self.num_sigma.setValue(10)
        blob_layout.addWidget(self.num_sigma)
        
        blob_group.setContentLayout(blob_layout)
        layout.addWidget(blob_group)
        layout.addSpacing(10)
        
        # Tracking Parameters
        tracking_group = self.create_collapsible_group("Tracking Parameters")
        tracking_layout = QVBoxLayout()
        
        # Max Age
        tracking_layout.addWidget(QLabel("Max Age:"))
        self.max_age = QSpinBox()
        self.max_age.setRange(1, 100)
        self.max_age.setValue(30)
        tracking_layout.addWidget(self.max_age)
        
        # N Init
        tracking_layout.addWidget(QLabel("N Init:"))
        self.n_init = QSpinBox()
        self.n_init.setRange(1, 10)
        self.n_init.setValue(3)
        tracking_layout.addWidget(self.n_init)
        
        # Max Cosine Distance
        tracking_layout.addWidget(QLabel("Max Cosine Distance:"))
        self.max_cosine_distance = QDoubleSpinBox()
        self.max_cosine_distance.setRange(0.01, 1.0)
        self.max_cosine_distance.setValue(0.2)
        self.max_cosine_distance.setSingleStep(0.01)
        tracking_layout.addWidget(self.max_cosine_distance)
        
        tracking_group.setContentLayout(tracking_layout)
        layout.addWidget(tracking_group)
        layout.addSpacing(10)
        
        # Output Parameters
        output_group = self.create_collapsible_group("Output Parameters")
        output_layout = QVBoxLayout()
        
        # Top N Cells
        output_layout.addWidget(QLabel("Top N Cells:"))
        self.top_n_cells = QSpinBox()
        self.top_n_cells.setRange(1, 50)
        self.top_n_cells.setValue(15)
        output_layout.addWidget(self.top_n_cells)
        
        # Max Frames
        output_layout.addWidget(QLabel("Max Frames (0 = All):"))
        self.max_frames = QSpinBox()
        self.max_frames.setRange(0, 10000)
        self.max_frames.setValue(0)
        self.max_frames.setSpecialValueText("All")
        output_layout.addWidget(self.max_frames)
        
        output_group.setContentLayout(output_layout)
        layout.addWidget(output_group)
        
        # Add stretch to push buttons to bottom
        layout.addStretch()
        
        # Process button
        self.process_button = QPushButton("Process")
        self.process_button.setMinimumHeight(36)
        self.process_button.clicked.connect(self.run_pipeline)
        layout.addWidget(self.process_button)
        
        # Save results button
        self.save_button = QPushButton("Save Results")
        self.save_button.setEnabled(False)
        self.save_button.setMinimumHeight(36)
        self.save_button.clicked.connect(self.save_results)
        layout.addWidget(self.save_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

    def toggle_kalman_options(self, state):
        """Show/hide Kalman filter parameters based on checkbox state"""
        self.kalman_params_widget.setVisible(state == Qt.CheckState.Checked.value)

    def run_pipeline(self):
        """Run the processing pipeline"""
        if not hasattr(self, 'input_file') or not self.input_file:
            QMessageBox.warning(self, "Warning", "Please select an input file first.")
            return
        
        # Check for ImageJ requirement
        if self.use_imagej.isChecked() and not IMAGEJ_AVAILABLE:
            QMessageBox.warning(self, "Warning", 
                              "ImageJ processing is not available. Please install pyimagej first.")
            self.use_imagej.setChecked(False)
            return
            
        # Disable UI elements during processing
        self.process_button.setEnabled(False)
        self.file_button.setEnabled(False)
        
        # Create parameters
        params = ProcessingParameters(
            blob_params=BlobParameters(
                min_sigma=self.min_sigma.value(),
                max_sigma=self.max_sigma.value(),
                threshold=self.threshold.value(),
                num_sigma=self.num_sigma.value()
            ),
            tracker_params=TrackerParameters(
                max_age=self.max_age.value(),
                n_init=self.n_init.value(),
                max_cosine_distance=self.max_cosine_distance.value()
            ),
            top_n_cells=self.top_n_cells.value(),
            max_frames=self.max_frames.value() if self.max_frames.value() > 0 else None
        )
        
        # Run in separate thread to avoid UI freezing
        self.processing_thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(self.input_file, params)
        )
        self.processing_thread.start()

    def _run_pipeline_thread(self, input_file, params):
        """Run pipeline in separate thread"""
        try:
            self.processed_frames, self.output_csv = self.pipeline.process_pipeline(
                input_file, 
                params,
                use_kalman=self.use_kalman.isChecked(),
                use_imagej=self.use_imagej.isChecked(),
                acquisition_noise=self.acquisition_noise.value(),
                kalman_bias=self.kalman_bias.value()
            )
            
            # Load tracking data into metrics viewer
            tracking_data = pd.read_csv(self.output_csv)
            
            # Create event to update UI
            event = QEvent(QEvent.Type.User)
            event.tracking_data = tracking_data  # Add tracking data to event
            QApplication.instance().postEvent(self, event)
        except Exception as e:
            event = QEvent(QEvent.Type.User + 1)
            event.error = str(e)
            QApplication.instance().postEvent(self, event)

    def select_file(self):
        """Open file dialog to select input TIFF"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select TIFF File", "", "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.input_file = file_path
            self.status_label.setText(f"Selected file: {Path(file_path).name}")
            self.process_button.setEnabled(True)

    def save_results(self):
        """Save processed results"""
        if not hasattr(self, 'processed_frames') or not self.processed_frames or not hasattr(self, 'output_csv') or not self.output_csv:
            QMessageBox.warning(self, "Warning", "No processed data to save.")
            return
        
        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return
        
        try:
            # Determine files to save
            files_to_save = []
            output_dir_path = Path(output_dir)
            
            # Save Kalman filtered output
            if self.use_kalman.isChecked():
                kalman_file = self.pipeline.temp_dir / "kalman_filtered.tiff"
                if kalman_file.exists():
                    target_kalman = output_dir_path / f"kalman_filtered_{Path(self.input_file).stem}.tiff"
                    shutil.copy(kalman_file, target_kalman)
                    files_to_save.append(str(target_kalman))
            
            # Save ImageJ processed output
            if self.use_imagej.isChecked():
                imagej_file = self.pipeline.temp_dir / "imagej_processed.tiff"
                if imagej_file.exists():
                    target_imagej = output_dir_path / f"imagej_processed_{Path(self.input_file).stem}.tiff"
                    shutil.copy(imagej_file, target_imagej)
                    files_to_save.append(str(target_imagej))
            
            # Save annotated TIFF
            source_tiff = self.pipeline.temp_dir / "annotated_output.tiff"
            if source_tiff.exists():
                target_tiff = output_dir_path / f"annotated_{Path(self.input_file).stem}.tiff"
                shutil.copy(source_tiff, target_tiff)
                files_to_save.append(str(target_tiff))
            
            # Save CSV data
            source_csv = Path(self.output_csv)
            if source_csv.exists():
                target_csv = output_dir_path / f"tracked_cells_{Path(self.input_file).stem}.csv"
                shutil.copy(source_csv, target_csv)
                files_to_save.append(str(target_csv))
            
            # Show success message with list of saved files
            message = f"Results saved to {output_dir}:\n" + "\n".join([f"- {Path(f).name}" for f in files_to_save])
            QMessageBox.information(self, "Success", message)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")

    def on_metrics_cell_selected(self, index):
        """Handle cell selection from metrics tab"""
        if index >= 0 and self.metrics_cell_combo.count() > 0:
            cell_id = int(self.metrics_cell_combo.itemText(index).split()[1])
            self.update_cell_metrics(cell_id)
            
            # Sync with tracking tab cell combo
            tracking_index = self.cell_combo.findText(f"Cell {cell_id}")
            if tracking_index >= 0:
                self.cell_combo.blockSignals(True)
                self.cell_combo.setCurrentIndex(tracking_index)
                self.cell_combo.blockSignals(False)

    def update_cell_metrics(self, cell_id):
        """Update all metrics displays for the selected cell"""
        if hasattr(self.metrics_viewer, 'df') and self.metrics_viewer.df is not None:
            # Update metrics viewer
            self.metrics_viewer.current_cell_id = cell_id
            self.metrics_viewer.calculate_full_metrics()
            
            # Update tracking tab metrics if exists
            if hasattr(self, 'tracking_tab_metrics') and hasattr(self.tracking_tab_metrics, 'df'):
                self.tracking_tab_metrics.current_cell_id = cell_id
                self.tracking_tab_metrics.calculate_full_metrics()
            
            # Update image viewer with cell data
            if hasattr(self.viewer, 'update_cell_data'):
                self.viewer.update_cell_data(self.metrics_viewer.full_data)
            
            # Update all plots
            current_frame = self.viewer.frame_slider.value() + 1
            self.update_all_plots(current_frame)

    def update_all_plots(self, frame_number):
        """Update all plots for the current frame"""
        if not hasattr(self.metrics_viewer, 'full_data') or not self.metrics_viewer.full_data['time']:
            return
        
        # Find the index for the current frame in the metrics data
        try:
            current_idx = self.metrics_viewer.full_data['time'].index(frame_number)
        except ValueError:
            return
        
        # Update tracking tab plots
        if hasattr(self, 'tracking_tab_metrics'):
            self.tracking_tab_metrics.update_plots(current_idx)
        
        # Update metrics tab plots
        self.metrics_viewer.update_plots(current_idx)
        
        # Update main viewer plots if it has the method
        if hasattr(self.viewer, 'update_plots_for_frame'):
            self.viewer.update_plots_for_frame(frame_number)

    def on_frame_changed(self, frame_number):
        """Handle frame changes from viewer"""
        # Update metrics displays
        if hasattr(self.metrics_viewer, 'update_frame'):
            self.metrics_viewer.update_frame(frame_number)
        
        if hasattr(self, 'tracking_tab_metrics'):
            self.tracking_tab_metrics.update_frame(frame_number)
        
        # Update all plots for the current frame
        self.update_all_plots(frame_number)

    def update_progress(self, percent, message):
        """Update progress bar and status label"""
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def event(self, event):
        """Handle custom events from worker thread"""
        if event.type() == QEvent.Type.User:
            # Processing complete
            if hasattr(self, 'processed_frames') and self.processed_frames:
                self.viewer.load_frames(self.processed_frames)
            
            # Use either the event.tracking_data or self.output_csv approach
            if hasattr(event, 'tracking_data'):
                tracking_data = event.tracking_data
            elif hasattr(self, 'output_csv') and self.output_csv:
                tracking_data = pd.read_csv(self.output_csv)
            else:
                tracking_data = None
                
            if tracking_data is not None:
                # Load tracking data into cell combos
                cell_ids = sorted(tracking_data['Cell ID'].unique())
                
                self.cell_combo.clear()
                self.metrics_cell_combo.clear()
                
                for cell_id in cell_ids:
                    self.cell_combo.addItem(f"Cell {cell_id}")
                    self.metrics_cell_combo.addItem(f"Cell {cell_id}")
                
                # Load data into metrics viewer
                self.metrics_viewer.load_data(tracking_data)
                # Also load into the tracking tab metrics viewer if it exists
                if hasattr(self, 'tracking_tab_metrics'):
                    self.tracking_tab_metrics.load_data(tracking_data)
                
                # Trigger initial cell selection
                if self.cell_combo.count() > 0:
                    self.cell_combo.setCurrentIndex(0)
                    self.on_cell_selected(0)
            
            self.save_button.setEnabled(True)
            self.file_button.setEnabled(True)
            self.process_button.setEnabled(True)
            return True
            
        elif event.type() == QEvent.Type.User + 1:
            # Error occurred
            QMessageBox.critical(self, "Error", getattr(event, "error", "Unknown error"))
            self.file_button.setEnabled(True)
            self.process_button.setEnabled(True)
            return True
        
        return super().event(event)