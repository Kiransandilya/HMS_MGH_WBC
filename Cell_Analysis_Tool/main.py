import sys
import os
import math
import random
import pyqtgraph as pg
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QFrame, QStackedWidget, QSizePolicy,
    QGraphicsOpacityEffect, QSplitter, QTabWidget, QGridLayout, QComboBox
)
from PyQt6.QtCore import (Qt, QSize, QPropertyAnimation, QEasingCurve, QTimer, QPointF, 
                        QVariantAnimation, QSequentialAnimationGroup, QParallelAnimationGroup,
                        pyqtProperty, QObject)
from PyQt6.QtGui import (QIcon, QPixmap, QColor, QPalette, QFont, QFontDatabase, QPainter, 
                        QBrush, QPen, QLinearGradient, QRadialGradient, QPainterPath, 
                        QTransform, QFontMetrics)


# Import other modules as in your original code
try:
    from complete_cell_analysis import CompleteCellAnalysisScreen
except ImportError:
    print("Warning: CompleteCellAnalysisScreen not found, complete cell analysis will be disabled.")
    CompleteCellAnalysisScreen = None


class StyleSheet:
    """Style constants for dark and light themes in Apple style"""
    
    # Dark Theme - using Apple's color palette
    DARK = {
        "bg_primary": "#1d1d1f",
        "bg_secondary": "#2d2d30",
        "accent": "#0071e3",
        "accent_secondary": "#42a1ec",
        "text_primary": "#ffffff",
        "text_secondary": "#bbbbbb",
        "border": "#3d3d3d",
        "button_hover": "#3d3d3d",
        "success": "#28c941",
        "warning": "#febc2e",
        "error": "#ff5f57",
        "title_bar": "#252528"
    }
    
    # Light Theme - using Apple's color palette
    LIGHT = {
        "bg_primary": "#f5f5f7",
        "bg_secondary": "#ffffff",
        "accent": "#0071e3",
        "accent_secondary": "#42a1ec",
        "text_primary": "#1d1d1f",
        "text_secondary": "#86868b",
        "border": "#dedede",
        "button_hover": "#f0f0f2",
        "success": "#28c941",
        "warning": "#febc2e",
        "error": "#ff5f57",
        "title_bar": "#f9f9fb"
    }


class ThemeToggle(QWidget):
    """Sophisticated toggle switch for theme selection"""
    
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.setFixedSize(26, 16)
        
        # State
        self._is_checked = dark_mode
        self._thumb_position = 18 if dark_mode else 8  # Start position
        
        # Animation
        self._animation = QPropertyAnimation(self, b"thumb_position")
        self._animation.setEasingCurve(QEasingCurve.Type.OutQuint)
        self._animation.setDuration(300)
    
    @pyqtProperty(float)
    def thumb_position(self):
        return self._thumb_position
    
    @thumb_position.setter
    def thumb_position(self, pos):
        self._thumb_position = pos
        self.update()
    
    def toggle(self):
        self._is_checked = not self._is_checked
        self._animation.stop()
        self._animation.setStartValue(self._thumb_position)
        
        if self._is_checked:  # Dark theme
            self._animation.setEndValue(18)
        else:  # Light theme
            self._animation.setEndValue(8)
            
        self._animation.start()
        
        # Notify parent
        if hasattr(self.parent(), 'toggle_theme'):
            self.parent().toggle_theme(self._is_checked)
    
    def mousePressEvent(self, event):
        self.toggle()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        # Draw track
        painter.setBrush(QBrush(QColor(theme["text_primary"]).darker() if self.dark_mode else QColor(theme["text_primary"])))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setOpacity(0.1)  # Subtle track
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 8, 8)
        painter.setOpacity(1.0)
        
        # Draw thumb
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.setPen(QPen(QColor(0, 0, 0, 20), 1))  # Very subtle border
        
        # Subtle shadow (manually drawn for better control)
        shadow_opacity = painter.opacity()
        painter.setOpacity(0.1)
        painter.setBrush(QBrush(QColor(0, 0, 0)))
        painter.drawEllipse(QPointF(self._thumb_position, 8), 6.5, 6.5)
        painter.setOpacity(shadow_opacity)
        
        # Draw actual thumb
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(QPointF(self._thumb_position, 8), 6, 6)
        
        # Draw icons on thumb
        if self._is_checked:  # Moon for dark theme
            # Draw crescent moon
            painter.save()
            painter.translate(self._thumb_position, 8)
            painter.scale(0.6, 0.6)
            
            moonPath = QPainterPath()
            moonPath.moveTo(3, -0.5)
            moonPath.arcTo(-2, -3, 10, 6, 70, 180)
            moonPath.arcTo(0, -1.5, 6, 3, 250, 180)
            moonPath.closeSubpath()
            
            painter.setBrush(QBrush(QColor(theme["bg_primary"])))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(moonPath)
            painter.restore()
        else:  # Sun for light theme
            # Draw sun
            painter.save()
            painter.translate(self._thumb_position, 8)
            painter.scale(0.4, 0.4)
            
            painter.setBrush(QBrush(QColor(theme["warning"])))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(0, 0), 4, 4)
            
            # Draw rays
            painter.setPen(QPen(QColor(theme["warning"]), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            ray_positions = 8
            ray_length = 3
            
            for i in range(ray_positions):
                angle = i * (360 / ray_positions)
                rad_angle = math.radians(angle)
                start_x = 5 * math.cos(rad_angle)
                start_y = 5 * math.sin(rad_angle)
                end_x = (5 + ray_length) * math.cos(rad_angle)
                end_y = (5 + ray_length) * math.sin(rad_angle)
                painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
            
            painter.restore()


class Cell:
    """Class to represent a cell with all its properties"""
    
    def __init__(self, position, radius, theme, is_main=False):
        self.position = position
        self.base_radius = radius
        self.current_radius = radius
        
        # Create gradient color for cell
        self.gradient = QRadialGradient(position, radius)
        self.gradient.setColorAt(0, QColor(theme["accent"]))
        self.gradient.setColorAt(1, QColor(theme["accent_secondary"]))
        
        self.opacity = random.uniform(0.5, 0.9) if not is_main else 0.9
        
        # Animation properties
        self.radius_variation = random.uniform(2, 5) if not is_main else 4
        self.radius_speed = random.uniform(0.001, 0.003)
        self.radius_phase = random.uniform(0, 2 * math.pi)
        
        self.opacity_variation = random.uniform(0.1, 0.3) if not is_main else 0.2
        self.opacity_speed = random.uniform(0.0005, 0.002)
        self.opacity_phase = random.uniform(0, 2 * math.pi)
        
        # Nucleus properties
        if is_main:
            self.has_nucleus = True
            self.nucleus_radius = radius * 0.36
            nucleus_offset = radius * 0.14
            self.nucleus_position = QPointF(
                position.x() + random.uniform(-nucleus_offset, nucleus_offset),
                position.y() + random.uniform(-nucleus_offset, nucleus_offset)
            )
            self.nucleus_color = QColor(0, 93, 179)
            self.nucleus_opacity = 0.8
            
            # Organelles only for main cell
            self.organelles = []
            for _ in range(random.randint(3, 5)):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0.3, 0.7) * radius
                self.organelles.append({
                    "position": QPointF(
                        position.x() + distance * math.cos(angle),
                        position.y() + distance * math.sin(angle)
                    ),
                    "radius": random.uniform(0.1, 0.15) * radius,
                    "color": QColor(255, 255, 255, random.randint(40, 60))
                })
        else:
            self.has_nucleus = random.random() > 0.2
            if self.has_nucleus:
                self.nucleus_radius = radius * random.uniform(0.3, 0.4)
                nucleus_offset = radius * 0.2
                self.nucleus_position = QPointF(
                    position.x() + random.uniform(-nucleus_offset, nucleus_offset),
                    position.y() + random.uniform(-nucleus_offset, nucleus_offset)
                )
                self.nucleus_color = QColor(0, 93, 179)
                self.nucleus_opacity = random.uniform(0.6, 0.8)
            
            # Smaller cells have fewer/no organelles
            self.organelles = []
            if radius > 20:
                for _ in range(random.randint(1, 3)):
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(0.3, 0.6) * radius
                    self.organelles.append({
                        "position": QPointF(
                            position.x() + distance * math.cos(angle),
                            position.y() + distance * math.sin(angle)
                        ),
                        "radius": random.uniform(0.08, 0.12) * radius,
                        "color": QColor(255, 255, 255, random.randint(30, 50))
                    })
        
        # Orbital motion for non-main cells
        if not is_main:
            self.orbit_radius_x = random.uniform(80, 120)
            self.orbit_radius_y = random.uniform(50, 90) 
            self.orbit_speed = random.uniform(0.0002, 0.0005)
            self.orbit_phase = random.uniform(0, 2 * math.pi)
            self.original_position = QPointF(position)
    
    def update(self, time_delta):
        """Update cell properties based on time"""
        # Radius animation
        self.current_radius = self.base_radius + self.radius_variation * math.sin(
            self.radius_phase + time_delta * self.radius_speed
        )
        
        # Opacity animation
        self.opacity = max(0.3, min(0.95, 
            self.opacity + self.opacity_variation * math.sin(
                self.opacity_phase + time_delta * self.opacity_speed
            )
        ))
        
        # Update nucleus properties if present
        if self.has_nucleus:
            self.nucleus_radius = self.current_radius * 0.36 + self.radius_variation * 0.1 * math.sin(
                self.radius_phase + time_delta * self.radius_speed * 1.5
            )
        
        # Orbital motion for non-main cells
        if hasattr(self, 'orbit_radius_x'):
            angle = self.orbit_phase + time_delta * self.orbit_speed
            self.position = QPointF(
                self.original_position.x() + self.orbit_radius_x * math.cos(angle),
                self.original_position.y() + self.orbit_radius_y * math.sin(angle)
            )


class CellVisualization(QWidget):
    """Sophisticated cell visualization with smooth animation"""
    
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.setMinimumSize(300, 300)
        
        # Animation properties
        self.animation_time = 0
        self.cells = []
        self.initialize_cells()
        
        # Start animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)  # ~60fps
    
    def initialize_cells(self):
        """Create cells with realistic properties"""
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        # Main central cell
        center = QPointF(self.width()/2, self.height()/2)
        self.cells.append(Cell(center, 70, theme, is_main=True))
        
        # Orbital cells
        for _ in range(4):
            size = random.uniform(16, 28)
            self.cells.append(Cell(center, size, theme))
    
    def update_theme(self, dark_mode):
        """Update the theme for all cells"""
        self.dark_mode = dark_mode
        theme = StyleSheet.DARK if dark_mode else StyleSheet.LIGHT
        
        # Recreate cells with new theme
        center = QPointF(self.width()/2, self.height()/2)
        self.cells = []
        self.cells.append(Cell(center, 70, theme, is_main=True))
        
        for _ in range(4):
            size = random.uniform(16, 28)
            self.cells.append(Cell(center, size, theme))
        
        self.update()
    
    def update_animation(self):
        """Update animation state and redraw"""
        self.animation_time += 16  # milliseconds since last update
        
        for cell in self.cells:
            cell.update(self.animation_time)
        
        self.update()
    
    def resizeEvent(self, event):
        """Handle resize by repositioning cells"""
        if not self.cells:
            return
            
        center = QPointF(self.width()/2, self.height()/2)
        
        # Update main cell position
        self.cells[0].position = center
        self.cells[0].original_position = QPointF(center)
        
        # Update orbital cells to orbit around the new center
        for i in range(1, len(self.cells)):
            if hasattr(self.cells[i], 'original_position'):
                self.cells[i].original_position = QPointF(center)
        
        super().resizeEvent(event)
    
    def paintEvent(self, event):
        """Draw the cells with sophisticated rendering"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        # Sort cells by size so smaller ones appear on top
        sorted_cells = sorted(self.cells, key=lambda c: c.current_radius, reverse=True)
        
        # Draw using frosted glass effect
        for cell in sorted_cells:
            # Draw cell body with gradient
            painter.save()
            painter.setOpacity(cell.opacity)
            
            # Create a radial gradient for the cell fill
            gradient = QRadialGradient(cell.position, cell.current_radius)
            theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
            gradient.setColorAt(0, QColor(theme["accent"]))
            gradient.setColorAt(1, QColor(theme["accent_secondary"]))
            
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            
            # Apply a blur for the frosted glass look
            blur_path = QPainterPath()
            blur_path.addEllipse(cell.position, cell.current_radius, cell.current_radius)
            painter.drawPath(blur_path)
            
            painter.restore()
            
            # Draw nucleus if present
            if cell.has_nucleus:
                painter.save()
                painter.setOpacity(cell.nucleus_opacity)
                painter.setBrush(QBrush(cell.nucleus_color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(cell.nucleus_position, cell.nucleus_radius, cell.nucleus_radius)
                painter.restore()
            
            # Draw organelles
            for organelle in cell.organelles:
                painter.save()
                painter.setOpacity(0.7 * cell.opacity)
                painter.setBrush(QBrush(organelle["color"]))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(organelle["position"], organelle["radius"], organelle["radius"])
                painter.restore()


class ElegantButton(QPushButton):
    """Styled button with Apple-like appearance"""
    
    def __init__(self, text, parent=None, dark_mode=True, is_primary=False, icon=None):
        super().__init__(text, parent)
        self.dark_mode = dark_mode
        self.is_primary = is_primary
        self.icon_name = icon
        self.icon_color = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(40)
        self.update_style()
    
    def update_style(self):
        """Update button style based on theme and type"""
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        if self.is_primary:
            # Primary button with gradient background
            gradient = f"qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {theme['accent']}, stop:1 {theme['accent_secondary']})"
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {gradient};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif;
                    font-weight: 500;
                    font-size: 15px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                              stop:0 {QColor(theme['accent']).lighter(110).name()}, 
                                              stop:1 {QColor(theme['accent_secondary']).lighter(110).name()});
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                              stop:0 {QColor(theme['accent']).darker(110).name()}, 
                                              stop:1 {QColor(theme['accent_secondary']).darker(110).name()});
                }}
            """)
        else:
            # Secondary button with subtle styling
            if self.dark_mode:
                self.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {theme['bg_secondary']};
                        color: {theme['text_primary']};
                        border: 1px solid {theme['border']};
                        border-radius: 8px;
                        padding: 10px 20px;
                        font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif;
                        font-weight: 500;
                        font-size: 15px;
                    }}
                    QPushButton:hover {{
                        background-color: {theme['button_hover']};
                    }}
                    QPushButton:pressed {{
                        background-color: {QColor(theme['button_hover']).darker(110).name()};
                    }}
                """)
            else:
                self.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {theme['bg_secondary']};
                        color: {theme['text_primary']};
                        border: 1px solid {theme['border']};
                        border-radius: 8px;
                        padding: 10px 20px;
                        font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif;
                        font-weight: 500;
                        font-size: 15px;
                    }}
                    QPushButton:hover {{
                        background-color: {theme['button_hover']};
                    }}
                    QPushButton:pressed {{
                        background-color: {QColor(theme['button_hover']).darker(105).name()};
                    }}
                """)
        
        # Set icon if provided
        if self.icon_name:
            self.update_icon()
    
    def update_icon(self):
        """Create and apply custom icon"""
        if not self.icon_name:
            return
            
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        icon_color = QColor(theme["accent"]) if not self.is_primary else QColor(theme["text_primary"])
        
        if self.icon_color != icon_color:
            self.icon_color = icon_color
            
            # Create icons based on name
            icon_size = QSize(24, 24)
            pixmap = QPixmap(icon_size)
            pixmap.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(self.icon_color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            
            if self.icon_name == "file":
                # File icon
                painter.drawRect(4, 3, 16, 18)
                painter.drawLine(8, 8, 16, 8)
                painter.drawLine(8, 12, 16, 12)
                painter.drawLine(8, 16, 13, 16)
            elif self.icon_name == "folder":
                # Folder icon
                painter.drawLine(4, 9, 8, 9)
                painter.drawLine(8, 9, 10, 6)
                painter.drawLine(10, 6, 20, 6)
                painter.drawLine(20, 6, 20, 18)
                painter.drawLine(20, 18, 4, 18)
                painter.drawLine(4, 18, 4, 9)
            elif self.icon_name == "output":
                # Output folder with arrow
                painter.drawLine(4, 9, 8, 9)
                painter.drawLine(8, 9, 10, 6)
                painter.drawLine(10, 6, 20, 6)
                painter.drawLine(20, 6, 20, 18)
                painter.drawLine(20, 18, 4, 18)
                painter.drawLine(4, 18, 4, 9)
                
                # Output arrow
                painter.drawLine(12, 12, 18, 12)
                painter.drawLine(15, 9, 18, 12)
                painter.drawLine(15, 15, 18, 12)
            elif self.icon_name == "back":
                # Back arrow
                painter.drawLine(12, 12, 6, 12)
                painter.drawLine(6, 12, 9, 9)
                painter.drawLine(6, 12, 9, 15)
            
            painter.end()
            
            icon = QIcon(pixmap)
            self.setIcon(icon)
            self.setIconSize(QSize(20, 20))
            
            # Set text alignment for icon buttons
            self.setStyleSheet(self.styleSheet() + """
                QPushButton {
                    text-align: left;
                    padding-left: 40px;
                }
            """)


class WelcomeScreen(QWidget):
    """Welcome screen with cell visualization and dive-in button"""
    
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.parent_app = parent  # Store reference to parent app
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 40, 0, 40)
        layout.setSpacing(20)
        
        # Theme toggle in top right
        self.theme_toggle = ThemeToggle(self, self.dark_mode)
        toggle_layout = QHBoxLayout()
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.theme_toggle)
        toggle_layout.setContentsMargins(0, 0, 20, 0)
        layout.addLayout(toggle_layout)
        
        # Cell visualization
        self.cell_viz = CellVisualization(self, self.dark_mode)
        layout.addWidget(self.cell_viz)
        
        # Welcome text
        self.title_label = QLabel("Welcome to Cell Analysis", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()  # Use system font instead of custom font
        font.setPointSize(24)
        font.setWeight(QFont.Weight.Medium)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)
        
        self.subtitle_label = QLabel("Your advanced tool for cellular research", self)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont()  # Use system font
        subtitle_font.setPointSize(15) 
        self.subtitle_label.setFont(subtitle_font)
        layout.addWidget(self.subtitle_label)
        
        # Dive-in button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.dive_in_button = ElegantButton("Dive In", self, self.dark_mode, is_primary=True)
        self.dive_in_button.setFixedSize(150, 45)
        self.dive_in_button.clicked.connect(self.on_dive_in_clicked)
        
        button_layout.addWidget(self.dive_in_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.update_theme()
    
    def toggle_theme(self, dark_mode):
        self.dark_mode = dark_mode
        self.update_theme()
        # Propagate to parent
        if hasattr(self.parent(), 'toggle_theme'):
            self.parent().toggle_theme(dark_mode)
    
    def update_theme(self):
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        # Update background
        self.setStyleSheet(f"background-color: {theme['bg_primary']};")
        
        # Update labels
        self.title_label.setStyleSheet(f"color: {theme['text_primary']};")
        self.subtitle_label.setStyleSheet(f"color: {theme['text_secondary']};")
        
        # Update cell visualization
        self.cell_viz.update_theme(self.dark_mode)
        
        # Update button
        self.dive_in_button.dark_mode = self.dark_mode
        self.dive_in_button.update_style()
    
    def on_dive_in_clicked(self):
        print("Dive In button clicked!")  # Debug print
        # Directly access the method from parent
        if self.parent_app is not None and hasattr(self.parent_app, 'show_analysis_type_screen'):
            print("Showing analysis type screen...")  # Debug print
            self.parent_app.show_analysis_type_screen()
        else:
            print("Failed to navigate: parent app not accessible or missing method")  # Debug print


class AnalysisOption(QFrame):
    """Custom widget for analysis type option"""
    
    def __init__(self, title, description, icon_type, parent=None, dark_mode=True, selected=False):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.selected = selected
        self.title = title
        self.description = description
        self.icon_type = icon_type
        
        self.init_ui()
    
    def init_ui(self):
        self.setFixedSize(240, 90)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(15)
        
        # Use QLabel for icon instead of QFrame - FIXED
        self.icon_label = QLabel(self)
        self.icon_label.setFixedSize(30, 30)
        layout.addWidget(self.icon_label)
        
        # Text container
        text_layout = QVBoxLayout()
        text_layout.setSpacing(5)
        
        self.title_label = QLabel(self.title, self)
        title_font = QFont()  # Use system font
        title_font.setPointSize(16)
        title_font.setWeight(QFont.Weight.Medium)
        self.title_label.setFont(title_font)
        
        self.description_label = QLabel(self.description, self)
        desc_font = QFont()  # Use system font
        desc_font.setPointSize(13)
        self.description_label.setFont(desc_font)
        
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.description_label)
        layout.addLayout(text_layout)
        
        self.update_style()
    
    def update_style(self):
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        # Set button gradient or solid color based on selection state
        if self.selected:
            self.setStyleSheet(f"""
                AnalysisOption {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                stop:0 {theme['accent']}, 
                                stop:1 {theme['accent_secondary']});
                    border-radius: 10px;
                }}
            """)
            self.title_label.setStyleSheet(f"color: white;")
            self.description_label.setStyleSheet(f"color: rgba(255, 255, 255, 0.8);")
            icon_color = "#ffffff"
        else:
            self.setStyleSheet(f"""
                AnalysisOption {{
                    background-color: {theme['bg_secondary']};
                    border-radius: 10px;
                }}
            """)
            self.title_label.setStyleSheet(f"color: {theme['text_primary']};")
            self.description_label.setStyleSheet(f"color: {theme['text_secondary']};")
            icon_color = theme['accent']
        
        # Create icon pixmap
        icon_pixmap = QPixmap(30, 30)
        icon_pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(icon_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background circle - FIXED brush creation
        if self.selected:
            bgColor = QColor(255, 255, 255, 50)  # White with 50/255 alpha
            painter.setBrush(QBrush(bgColor))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(0, 0, 30, 30)
        else:
            bgColor = QColor(theme['accent'])
            bgColor.setAlpha(30)  # 30/255 alpha
            painter.setBrush(QBrush(bgColor))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(0, 0, 30, 30)
        
        # Draw icon based on type
        painter.setPen(QPen(QColor(icon_color), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        
        if self.icon_type == "manual":
            # Manual labeling icon (X shape)
            painter.drawLine(10, 10, 20, 20)
            painter.drawLine(10, 20, 20, 10)
        elif self.icon_type == "analysis":
            # Analysis icon (magnifying glass)
            painter.drawEllipse(8, 8, 14, 14)
            painter.drawLine(19, 19, 22, 22)
        elif self.icon_type == "advanced":
            # Advanced icon (plus sign)
            painter.drawLine(15, 8, 15, 22)
            painter.drawLine(8, 15, 22, 15)
        
        painter.end()
        
        # Set pixmap to QLabel - FIXED
        self.icon_label.setPixmap(icon_pixmap)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def mousePressEvent(self, event):
        if hasattr(self.parent(), 'select_option'):
            self.parent().select_option(self.title)


class AnalysisTypeScreen(QWidget):
    """Screen for selecting analysis type"""
    
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.parent_app = parent  # Store reference to parent app
        self.selected_option = "Complete Cell Analysis"  # Default selected option
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 40, 0, 40)
        layout.setSpacing(20)
        
        # Theme toggle in top right
        self.theme_toggle = ThemeToggle(self, self.dark_mode)
        toggle_layout = QHBoxLayout()
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.theme_toggle)
        toggle_layout.setContentsMargins(0, 0, 20, 0)
        layout.addLayout(toggle_layout)
        
        # Content layout with proper margins
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(30, 0, 30, 0)
        content_layout.setSpacing(30)
        
        # Title
        self.title_label = QLabel("Choose Analysis Type", self)
        font = QFont()  # Use system font
        font.setPointSize(20)
        font.setWeight(QFont.Weight.Medium)
        self.title_label.setFont(font)
        content_layout.addWidget(self.title_label)
        
        # Analysis options
        self.option_manual = AnalysisOption(
            "Manual Labelling", 
            "Label cell samples manually", 
            "manual", 
            self, 
            self.dark_mode,
            self.selected_option == "Manual Labelling"
        )
        content_layout.addWidget(self.option_manual)
        
        self.option_complete = AnalysisOption(
            "Complete Cell Analysis", 
            "Full automated analysis", 
            "analysis", 
            self, 
            self.dark_mode,
            self.selected_option == "Complete Cell Analysis"
        )
        content_layout.addWidget(self.option_complete)
        
        self.option_advanced = AnalysisOption(
            "Advanced Options", 
            "Additional analysis tools", 
            "advanced", 
            self, 
            self.dark_mode,
            self.selected_option == "Advanced Options"
        )
        content_layout.addWidget(self.option_advanced)
        
        # Back button
        button_layout = QHBoxLayout()
        self.back_button = ElegantButton("Back", self, self.dark_mode, is_primary=False, icon="back")
        self.back_button.setFixedSize(100, 40)
        self.back_button.clicked.connect(self.on_back_clicked)
        button_layout.addWidget(self.back_button)
        button_layout.addStretch()
        
        content_layout.addStretch()
        content_layout.addLayout(button_layout)
        
        layout.addLayout(content_layout)
        
        self.update_theme()
    
    def toggle_theme(self, dark_mode):
        self.dark_mode = dark_mode
        self.update_theme()
        # Propagate to parent
        if hasattr(self.parent(), 'toggle_theme'):
            self.parent().toggle_theme(dark_mode)
    
    def update_theme(self):
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        # Update background
        self.setStyleSheet(f"background-color: {theme['bg_primary']};")
        
        # Update title
        self.title_label.setStyleSheet(f"color: {theme['text_primary']};")
        
        # Update options
        self.option_manual.dark_mode = self.dark_mode
        self.option_manual.update_style()
        
        self.option_complete.dark_mode = self.dark_mode
        self.option_complete.update_style()
        
        self.option_advanced.dark_mode = self.dark_mode
        self.option_advanced.update_style()
        
        # Update button
        self.back_button.dark_mode = self.dark_mode
        self.back_button.update_style()
    
    def select_option(self, option_title):
        print(f"Option selected: {option_title}")  # Debug print
        self.selected_option = option_title
        
        # Update selection states
        self.option_manual.selected = (option_title == "Manual Labelling")
        self.option_manual.update_style()
        
        self.option_complete.selected = (option_title == "Complete Cell Analysis")
        self.option_complete.update_style()
        
        self.option_advanced.selected = (option_title == "Advanced Options")
        self.option_advanced.update_style()
        
        # If Manual Labelling is selected, show that screen
        if option_title == "Manual Labelling" and self.parent_app is not None and hasattr(self.parent_app, 'show_manual_labeling_screen'):
            print("Showing manual labeling screen...")  # Debug print
            self.parent_app.show_manual_labeling_screen()
        # If complete cell analysis is selected, proceed to that screen
        elif option_title == "Complete Cell Analysis" and self.parent_app is not None and hasattr(self.parent_app, 'show_analysis_screen'):
            print("Showing analysis screen...")  # Debug print
            self.parent_app.show_analysis_screen()
        
    def on_back_clicked(self):
        # Return to welcome screen
        if self.parent_app is not None and hasattr(self.parent_app, 'show_welcome_screen'):
            print("Going back to welcome screen...")  # Debug print
            self.parent_app.show_welcome_screen()


class FileSelectionButton(QWidget):
    """Custom button for file/directory selection"""
    
    def __init__(self, title, icon_type, parent=None, dark_mode=True):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.title = title
        self.icon_type = icon_type
        self.hovered = False
        
        self._clicked_callback = None
        
        self.init_ui()
    
    def init_ui(self):
        self.setFixedHeight(50)
        self.setMinimumWidth(240)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 5, 15, 5)
        layout.setSpacing(15)
        
        # Icon container
        self.icon_container = QLabel(self)
        self.icon_container.setFixedSize(24, 24)
        layout.addWidget(self.icon_container)
        
        # Title
        self.title_label = QLabel(self.title, self)
        title_font = QFont()  # Use system font
        title_font.setPointSize(15)
        title_font.setWeight(QFont.Weight.Medium)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        self.update_style()
    
    def update_style(self):
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        # Button background
        if self.hovered:
            self.setStyleSheet(f"""
                FileSelectionButton {{
                    background-color: {theme['bg_secondary']};
                    border-radius: 8px;
                    border: 1px solid {theme['accent']};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                FileSelectionButton {{
                    background-color: {theme['bg_secondary']};
                    border-radius: 8px;
                }}
            """)
        
        # Title color
        self.title_label.setStyleSheet(f"color: {theme['text_primary']};")
        
        # Icon
        icon_pixmap = QPixmap(24, 24)
        icon_pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(icon_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(theme['accent']), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        
        if self.icon_type == "file":
            # File icon
            painter.drawRect(4, 2, 16, 20)
            painter.drawLine(8, 8, 16, 8)
            painter.drawLine(8, 12, 16, 12)
            painter.drawLine(8, 16, 13, 16)
        elif self.icon_type == "folder":
            # Folder icon
            painter.drawLine(2, 7, 8, 7)
            painter.drawLine(8, 7, 10, 4)
            painter.drawLine(10, 4, 20, 4)
            painter.drawLine(20, 4, 20, 20)
            painter.drawLine(20, 20, 2, 20)
            painter.drawLine(2, 20, 2, 7)
        elif self.icon_type == "output":
            # Output folder with arrow
            painter.drawLine(2, 7, 8, 7)
            painter.drawLine(8, 7, 10, 4)
            painter.drawLine(10, 4, 20, 4)
            painter.drawLine(20, 4, 20, 20)
            painter.drawLine(20, 20, 2, 20)
            painter.drawLine(2, 20, 2, 7)
            
            # Output arrow
            painter.drawLine(10, 12, 16, 12)
            painter.drawLine(13, 9, 16, 12)
            painter.drawLine(13, 15, 16, 12)
        
        painter.end()
        
        self.icon_container.setPixmap(icon_pixmap)
    
    def enterEvent(self, event):
        self.hovered = True
        self.update_style()
    
    def leaveEvent(self, event):
        self.hovered = False
        self.update_style()
    
    def mousePressEvent(self, event):
        if self._clicked_callback:
            self._clicked_callback()
    
    def clicked(self, callback):
        """Set a callback function for when the button is clicked"""
        self._clicked_callback = callback


class AnalysisScreen(QWidget):
    """Screen for file selection and analysis"""
    
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.parent_app = parent  # Store reference to parent app
        self.selected_file = None
        self.selected_directory = None
        self.output_folder = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 40, 0, 40)
        layout.setSpacing(20)
        
        # Theme toggle in top right
        self.theme_toggle = ThemeToggle(self, self.dark_mode)
        toggle_layout = QHBoxLayout()
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.theme_toggle)
        toggle_layout.setContentsMargins(0, 0, 20, 0)
        layout.addLayout(toggle_layout)
        
        # Content layout with proper margins
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(30, 0, 30, 0)
        content_layout.setSpacing(15)
        
        # Title and subtitle
        self.title_label = QLabel("Complete Analysis", self)
        font = QFont()  # Use system font
        font.setPointSize(20)
        font.setWeight(QFont.Weight.Medium)
        self.title_label.setFont(font)
        content_layout.addWidget(self.title_label)
        
        self.subtitle_label = QLabel("Select Data Source", self)
        subtitle_font = QFont()  # Use system font
        subtitle_font.setPointSize(15)
        self.subtitle_label.setFont(subtitle_font)
        content_layout.addWidget(self.subtitle_label)
        content_layout.addSpacing(10)
        
        # File selection buttons
        self.file_button = FileSelectionButton("Select File", "file", self, self.dark_mode)
        self.file_button.clicked(self.on_select_file)
        content_layout.addWidget(self.file_button)
        
        self.directory_button = FileSelectionButton("Select Directory", "folder", self, self.dark_mode)
        self.directory_button.clicked(self.on_select_directory)
        content_layout.addWidget(self.directory_button)
        
        self.output_button = FileSelectionButton("Select Output Folder", "output", self, self.dark_mode)
        self.output_button.clicked(self.on_select_output)
        content_layout.addWidget(self.output_button)
        
        # File information frame
        self.file_info_frame = QFrame(self)
        self.file_info_frame.setMinimumHeight(120)
        self.file_info_frame.setVisible(False)
        
        file_info_layout = QVBoxLayout(self.file_info_frame)
        file_info_layout.setContentsMargins(20, 15, 20, 15)
        file_info_layout.setSpacing(10)
        
        self.info_title = QLabel("Selected Sample", self.file_info_frame)
        info_title_font = QFont()  # Use system font
        info_title_font.setPointSize(15)
        info_title_font.setWeight(QFont.Weight.Medium)
        self.info_title.setFont(info_title_font)
        file_info_layout.addWidget(self.info_title)
        
        # File path label with icon
        file_path_layout = QHBoxLayout()
        self.file_icon = QLabel(self.file_info_frame)
        self.file_icon.setFixedSize(24, 24)
        file_path_layout.addWidget(self.file_icon)
        
        self.file_path = QLabel(self.file_info_frame)
        file_path_font = QFont()  # Use system font
        file_path_font.setPointSize(12)
        self.file_path.setFont(file_path_font)
        file_path_layout.addWidget(self.file_path)
        file_path_layout.addStretch()
        
        file_info_layout.addLayout(file_path_layout)
        
        self.file_details = QLabel(self.file_info_frame)
        file_details_font = QFont()  # Use system font
        file_details_font.setPointSize(12)
        self.file_details.setFont(file_details_font)
        file_info_layout.addWidget(self.file_details)
        
        content_layout.addWidget(self.file_info_frame)
        content_layout.addStretch()
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.back_button = ElegantButton("Back", self, self.dark_mode, is_primary=False, icon="back")
        self.back_button.setFixedSize(100, 40)
        self.back_button.clicked.connect(self.on_back_clicked)
        button_layout.addWidget(self.back_button)
        
        button_layout.addStretch()
        
        self.analyze_button = ElegantButton("Analyze", self, self.dark_mode, is_primary=True)
        self.analyze_button.setFixedSize(100, 40)
        self.analyze_button.clicked.connect(self.on_analyze)
        button_layout.addWidget(self.analyze_button)
        
        content_layout.addLayout(button_layout)
        
        layout.addLayout(content_layout)
        
        self.update_theme()
    
    def toggle_theme(self, dark_mode):
        self.dark_mode = dark_mode
        self.update_theme()
        # Propagate to parent
        if hasattr(self.parent(), 'toggle_theme'):
            self.parent().toggle_theme(dark_mode)
    
    def update_theme(self):
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        # Update background
        self.setStyleSheet(f"background-color: {theme['bg_primary']};")
        
        # Update title and subtitle
        self.title_label.setStyleSheet(f"color: {theme['text_primary']};")
        self.subtitle_label.setStyleSheet(f"color: {theme['text_secondary']};")
        
        # Update selection buttons
        self.file_button.dark_mode = self.dark_mode
        self.file_button.update_style()
        
        self.directory_button.dark_mode = self.dark_mode
        self.directory_button.update_style()
        
        self.output_button.dark_mode = self.dark_mode
        self.output_button.update_style()
        
        # Update file info frame
        if self.dark_mode:
            self.file_info_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {theme['bg_secondary']};
                    border-radius: 10px;
                }}
            """)
        else:
            self.file_info_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {theme['bg_secondary']};
                    border: 1px solid {theme['border']};
                    border-radius: 10px;
                }}
            """)
        
        self.info_title.setStyleSheet(f"color: {theme['text_primary']};")
        self.file_path.setStyleSheet(f"color: {theme['text_primary']};")
        self.file_details.setStyleSheet(f"color: {theme['text_secondary']};")
        
        # Update file icon if visible
        if self.file_info_frame.isVisible():
            self.update_file_icon()
        
        # Update buttons
        self.back_button.dark_mode = self.dark_mode
        self.back_button.update_style()
        
        self.analyze_button.dark_mode = self.dark_mode
        self.analyze_button.update_style()
    
    def update_file_icon(self):
        """Update file icon based on theme and file type"""
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        icon_pixmap = QPixmap(24, 24)
        icon_pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(icon_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(theme['accent']), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        
        # Choose icon based on if file or directory was selected
        if self.selected_directory:
            # Folder icon
            painter.drawLine(2, 7, 8, 7)
            painter.drawLine(8, 7, 10, 4)
            painter.drawLine(10, 4, 20, 4)
            painter.drawLine(20, 4, 20, 20)
            painter.drawLine(20, 20, 2, 20)
            painter.drawLine(2, 20, 2, 7)
        else:
            # File icon
            painter.drawRect(4, 2, 16, 20)
            painter.drawLine(8, 8, 16, 8)
            painter.drawLine(8, 12, 16, 12)
            painter.drawLine(8, 16, 13, 16)
        
        painter.end()
        
        self.file_icon.setPixmap(icon_pixmap)
    
    def on_select_file(self):
        """Handle file selection"""
        options = QFileDialog.Option.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "All Files (*)", options=options
        )
        
        if file_path:
            self.selected_file = file_path
            self.selected_directory = None
            
            # Get file details
            file_info = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            
            # Update file info display
            self.file_path.setText(file_info)
            self.file_details.setText(f"{file_size:.1f} MB")
            
            # Show the file info frame
            self.file_info_frame.setVisible(True)
            self.update_file_icon()
    
    def on_select_directory(self):
        """Handle directory selection"""
        options = QFileDialog.Option.ReadOnly
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", "", options=options
        )
        
        if directory:
            self.selected_directory = directory
            self.selected_file = None
            
            try:
                # Count files in directory
                file_count = sum(1 for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))
                
                # Get directory size
                dir_size = 0
                for dirpath, dirnames, filenames in os.walk(directory):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            try:
                                dir_size += os.path.getsize(fp)
                            except (OSError, PermissionError):
                                pass  # Skip files we can't access
                
                dir_size_mb = dir_size / (1024 * 1024)  # Size in MB
                
                # Update file info display
                self.file_path.setText(os.path.basename(directory))
                self.file_details.setText(f"{file_count} files • {dir_size_mb:.1f} MB")
            except Exception as e:
                # Handle any errors accessing directory
                self.file_path.setText(os.path.basename(directory))
                self.file_details.setText("Error reading directory")
                print(f"Error reading directory: {str(e)}")
            
            # Show the file info frame
            self.file_info_frame.setVisible(True)
            self.update_file_icon()
    
    def on_select_output(self):
        """Handle output folder selection"""
        options = QFileDialog.Option.ReadOnly
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", "", options=options
        )
        
        if directory:
            self.output_folder = directory
    
    def on_back_clicked(self):
        """Return to analysis type screen"""
        if self.parent_app is not None and hasattr(self.parent_app, 'show_analysis_type_screen'):
            print("Going back to analysis type screen...")  # Debug print
            self.parent_app.show_analysis_type_screen()
    
    def on_analyze(self):
        """Start analysis with selected files"""
        if self.selected_file or self.selected_directory:
            # In a real app, this would trigger the analysis process
            # For now, just print what would be analyzed
            source = self.selected_file if self.selected_file else self.selected_directory
            output = self.output_folder if self.output_folder else "Default output location"
            
            print(f"Analyzing: {source}")
            print(f"Output to: {output}")
            
            # Here you would connect to your Python analysis code


class CellAnalysisApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.dark_mode = True  # Start with dark mode
        self.init_ui()
        self.has_complete_analysis = CompleteCellAnalysisScreen is not None
    
    def init_ui(self):
        # Set window properties
        self.setWindowTitle("Cell Analysis")
        self.setMinimumSize(1000, 700)
        
        # Create central stacked widget
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create screens - pass self as parent explicitly
        self.welcome_screen = WelcomeScreen(self, self.dark_mode)
        self.analysis_type_screen = AnalysisTypeScreen(self, self.dark_mode)
        self.analysis_screen = AnalysisScreen(self, self.dark_mode)
        
        # Add screens to stacked widget
        self.central_widget.addWidget(self.welcome_screen)
        self.central_widget.addWidget(self.analysis_type_screen)
        self.central_widget.addWidget(self.analysis_screen)
        
        # Start with welcome screen
        self.central_widget.setCurrentWidget(self.welcome_screen)
        
        # Apply theme
        self.apply_theme()
        
        print("Application initialized")  # Debug print

    def show_manual_labeling_screen(self):
        """Switch to manual labeling screen"""
        # Import here to avoid circular imports
        from manual_labeling import ManualLabelingScreen
        
        # Create screen if it doesn't exist yet
        if not hasattr(self, 'manual_labeling_screen') or self.manual_labeling_screen is None:
            self.manual_labeling_screen = ManualLabelingScreen(self, self.dark_mode)
            self.central_widget.addWidget(self.manual_labeling_screen)
        
        # Switch to the manual labeling screen
        self.central_widget.setCurrentWidget(self.manual_labeling_screen)
    
    def toggle_theme(self, dark_mode):
        """Toggle between dark and light themes"""
        self.dark_mode = dark_mode
        self.apply_theme()
    
    def apply_theme(self):
        """Apply current theme to all UI elements"""
        theme = StyleSheet.DARK if self.dark_mode else StyleSheet.LIGHT
        
        # Set application palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(theme["bg_primary"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(theme["text_primary"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(theme["bg_secondary"]))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme["bg_primary"]))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(theme["bg_secondary"]))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(theme["text_primary"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(theme["text_primary"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(theme["bg_secondary"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme["text_primary"]))
        palette.setColor(QPalette.ColorRole.Link, QColor(theme["accent"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(theme["accent"]))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        
        self.setPalette(palette)
        
        # Update screens
        self.welcome_screen.dark_mode = self.dark_mode
        self.welcome_screen.update_theme()
        
        self.analysis_type_screen.dark_mode = self.dark_mode
        self.analysis_type_screen.update_theme()
        
        if hasattr(self, 'complete_analysis_screen') and self.complete_analysis_screen is not None:
            self.complete_analysis_screen.dark_mode = self.dark_mode
            self.complete_analysis_screen.update_theme()

    
    def show_welcome_screen(self):
        """Switch to welcome screen"""
        print("Switching to welcome screen")  # Debug print
        self.central_widget.setCurrentWidget(self.welcome_screen)
    
    def show_analysis_type_screen(self):
        """Switch to analysis type selection screen"""
        print("Switching to analysis type screen")  # Debug print
        self.central_widget.setCurrentWidget(self.analysis_type_screen)
        
    def show_analysis_screen(self):
        """Switch to analysis screen"""
        print("Switching to analysis screen")  # Debug print
        
        # Check if we need to create the analysis screen
        if not hasattr(self, 'complete_analysis_screen') or self.complete_analysis_screen is None:
            # Create the screen if the class is available
            if CompleteCellAnalysisScreen is not None:
                self.complete_analysis_screen = CompleteCellAnalysisScreen(self, self.dark_mode)
                self.central_widget.addWidget(self.complete_analysis_screen)
            else:
                # Fall back to regular analysis screen if the class is not available
                self.central_widget.setCurrentWidget(self.analysis_screen)
                return
        
        # Switch to the complete analysis screen
        self.central_widget.setCurrentWidget(self.complete_analysis_screen)


def main():
    # Create application
    app = QApplication(sys.argv)
    
    # Set application-wide font (use system fonts instead of custom fonts)
    # This prevents the warning about SF Pro Display
    if sys.platform == "darwin":  # macOS
        font_family = ".AppleSystemUIFont"
    elif sys.platform == "win32":  # Windows
        font_family = "Segoe UI"
    else:  # Linux/other
        font_family = "Roboto"
    
    app.setFont(QFont(font_family, 12))
    
    # Create and show the application window
    window = CellAnalysisApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()