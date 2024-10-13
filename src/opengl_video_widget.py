from PyQt6.QtOpenGLWidgets import QOpenGLWidget  # Use QOpenGLWidget here
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt
import OpenGL.GL as gl
import numpy as np
import cv2


class OpenGLVideoWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.texture_id = None
        self.image = None
        self.bounding_boxes = []  # List to store bounding boxes

    def initializeGL(self):
        """Initialize OpenGL settings."""
        print("Initializing OpenGL...")  # Log initialization
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0, 0, 0, 1)  # Set clear color to black
        gl.glEnable(gl.GL_DEPTH_TEST)  # Enable depth testing if needed
        gl.glEnable(gl.GL_TEXTURE_2D)  # Ensure texturing is enabled
        # Query and print OpenGL info (Vendor, Renderer, and Version)
        vendor = gl.glGetString(gl.GL_VENDOR).decode('utf-8')
        renderer = gl.glGetString(gl.GL_RENDERER).decode('utf-8')
        version = gl.glGetString(gl.GL_VERSION).decode('utf-8')

        print(f"OpenGL Vendor: {vendor}")
        print(f"OpenGL Renderer: {renderer}")
        print(f"OpenGL Version: {version}")

    def resizeGL(self, width, height):
        """Adjust the viewport to maintain the aspect ratio of the video."""
        print(f"OpenGL widget resized to: {width}x{height}")

        if self.image is not None:
            # Get the aspect ratio of the video frame
            video_width, video_height = self.image.shape[1], self.image.shape[0]
            video_aspect = video_width / video_height

            # Get the aspect ratio of the available widget area
            widget_aspect = width / height

            # Calculate the scaling factor and the dimensions that respect the aspect ratio
            if widget_aspect > video_aspect:
                # Window is wider than the video aspect ratio
                scaled_width = int(height * video_aspect)
                scaled_height = height
            else:
                # Window is taller than the video aspect ratio
                scaled_width = width
                scaled_height = int(width / video_aspect)

            # Center the viewport in the available window
            x_offset = (width - scaled_width) // 2
            y_offset = (height - scaled_height) // 2

            # Set the viewport to the calculated dimensions
            gl.glViewport(x_offset, y_offset, scaled_width, scaled_height)
        else:
            # If no image is loaded, just fill the whole window
            gl.glViewport(0, 0, width, height)

    def paintGL(self):
        """Render the current frame and bounding boxes."""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Reset the color to white before drawing the texture
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)

        # Render the texture and bounding boxes together
        self.draw_texture_and_bounding_boxes()

        # Check for OpenGL errors
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"OpenGL Error: {error}")

    def upload_frame_to_opengl(self, frame):
        """Upload the captured frame to OpenGL as a texture."""
        self.image = frame
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # Convert to RGBA format

        if self.texture_id is None:
            self.texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, frame_rgba.shape[1], frame_rgba.shape[0], 0, gl.GL_RGBA,
                            gl.GL_UNSIGNED_BYTE, frame_rgba)
            # Set texture parameters (if not already set)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        else:
            # Only update the texture data without reallocating memory
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, frame_rgba.shape[1], frame_rgba.shape[0], gl.GL_RGBA,
                               gl.GL_UNSIGNED_BYTE, frame_rgba)

    def draw_texture_and_bounding_boxes(self):
        """Draw the uploaded texture (video frame) and bounding boxes on the screen."""
        if self.texture_id is None or self.image is None:
            return  # No texture to draw

        # Bind the texture for rendering
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

        # Get the aspect ratio of the video frame
        video_width, video_height = self.image.shape[1], self.image.shape[0]
        video_aspect = video_width / video_height

        # Get the aspect ratio of the widget
        widget_width, widget_height = self.width(), self.height()
        widget_aspect = widget_width / widget_height

        # Compute scaling factors and offsets to maintain aspect ratio
        if widget_aspect > video_aspect:
            # Widget is wider than the video, scale by height
            scale = widget_height / video_height
            scaled_width = video_width * scale
            scaled_height = widget_height
            x_offset = (widget_width - scaled_width) / 2
            y_offset = 0
        else:
            # Widget is taller than the video, scale by width
            scale = widget_width / video_width
            scaled_width = widget_width
            scaled_height = video_height * scale
            x_offset = 0
            y_offset = (widget_height - scaled_height) / 2

        # Set viewport based on calculated scaled dimensions and offsets
        gl.glViewport(int(x_offset), int(y_offset), int(scaled_width), int(scaled_height))

        # Draw the textured quad
        gl.glBegin(gl.GL_QUADS)

        # Use the scaled and centered texture coordinates
        gl.glTexCoord2f(0, 1)
        gl.glVertex3f(-1, -1, 0)  # Bottom left

        gl.glTexCoord2f(1, 1)
        gl.glVertex3f(1, -1, 0)  # Bottom right

        gl.glTexCoord2f(1, 0)
        gl.glVertex3f(1, 1, 0)  # Top right

        gl.glTexCoord2f(0, 0)
        gl.glVertex3f(-1, 1, 0)  # Top left

        gl.glEnd()

        # Unbind the texture to ensure correct rendering of subsequent elements
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Disable texturing and depth testing for drawing bounding boxes
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_DEPTH_TEST)

        # Draw bounding boxes
        if self.bounding_boxes:
            for box, score, class_id in self.bounding_boxes:
                x1, y1, x2, y2 = box

                # Scale bounding box coordinates to match the scaled video size
                x1 = x_offset + (x1 / video_width) * scaled_width
                y1 = y_offset + (y1 / video_height) * scaled_height
                x2 = x_offset + (x2 / video_width) * scaled_width
                y2 = y_offset + (y2 / video_height) * scaled_height

                # Normalize the coordinates to NDC
                x1_ndc = (x1 / widget_width) * 2 - 1
                y1_ndc = 1 - (y1 / widget_height) * 2
                x2_ndc = (x2 / widget_width) * 2 - 1
                y2_ndc = 1 - (y2 / widget_height) * 2

                # Set the color to red with 50% opacity for the filled square
                gl.glColor4f(1.0, 0.0, 0.0, 0.5)

                # Draw the filled bounding box
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex2f(x1_ndc, y1_ndc)
                gl.glVertex2f(x2_ndc, y1_ndc)
                gl.glVertex2f(x2_ndc, y2_ndc)
                gl.glVertex2f(x1_ndc, y2_ndc)
                gl.glEnd()

                # Set the color to red with full opacity for the outline
                gl.glColor4f(1.0, 0.0, 0.0, 1.0)

                # Draw the bounding box outline
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glVertex2f(x1_ndc, y1_ndc)
                gl.glVertex2f(x2_ndc, y1_ndc)
                gl.glVertex2f(x2_ndc, y2_ndc)
                gl.glVertex2f(x1_ndc, y2_ndc)
                gl.glEnd()

        # Re-enable texturing and depth testing for future rendering
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Check for OpenGL errors (useful for debugging)
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"OpenGL Error: {error}")



