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

    def resizeGL(self, width, height):
        """Adjust the viewport and projection."""
        print(f"OpenGL widget resized to: {width}x{height}")  # Log resize events
        gl.glViewport(0, 0, width, height)

    def paintGL(self):
        """Render the current frame and bounding boxes."""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Render the latest uploaded texture (video frame)
        self.draw_texture()  # Ensure draw_texture is called here

        # Draw the bounding boxes if they exist
        if self.bounding_boxes:
            self.draw_bounding_boxes()  # Call to draw bounding boxes

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

    def draw_texture(self):
        """Draw the uploaded texture (video frame) on the screen."""
        if self.texture_id is None:
            self.texture_id = gl.glGenTextures(1)

        # Bind the texture for rendering
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

        # Draw the textured quad
        gl.glBegin(gl.GL_QUADS)

        # Adjust texture coordinates to flip vertically
        gl.glTexCoord2f(0, 1); gl.glVertex3f(-1, -1, 0)        # Bottom left
        gl.glTexCoord2f(1, 1); gl.glVertex3f(1, -1, 0)      # Bottom right
        gl.glTexCoord2f(1, 0); gl.glVertex3f(1, 1, 0)    # Top right
        gl.glTexCoord2f(0, 0); gl.glVertex3f(-1, 1, 0)      # Top left
        gl.glEnd()

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)  # Unbind texture after drawing

    # TODO: Switched to NDC
    def draw_bounding_boxes(self):
        if self.image is None:
            print("No image available for drawing bounding boxes.")
            return  # Exit if image is not available

        """Draw the bounding boxes based on current detection results."""
        for (box, score, class_id) in self.bounding_boxes:
            print(f"Drawing box: {box} with score {score} for class {class_id}")
            color = (0.0, 1.0, 0.0, 0.5)  # Green with transparency for high confidence
            gl.glColor4f(color[0], color[1], color[2], color[3])  # Set the color with alpha

            x1, y1, x2, y2 = box
            # Convert to normalized device coordinates
            x1_ndc = (x1 / self.image.shape[1]) * 2 - 1
            y1_ndc = 1 - (y1 / self.image.shape[0]) * 2
            x2_ndc = (x2 / self.image.shape[1]) * 2 - 1
            y2_ndc = 1 - (y2 / self.image.shape[0]) * 2

            print(f"Converted NDC coordinates: ({x1_ndc}, {y1_ndc}), ({x2_ndc}, {y2_ndc})")

            # Ensure the OpenGL state is set correctly
            gl.glDisable(gl.GL_TEXTURE_2D)  # Disable texturing
            gl.glEnable(gl.GL_BLEND)  # Enable blending
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)  # Set blend function

            # Draw filled rectangle for the transparent overlay
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex2f(x1_ndc, y1_ndc)
            gl.glVertex2f(x2_ndc, y1_ndc)
            gl.glVertex2f(x2_ndc, y2_ndc)
            gl.glVertex2f(x1_ndc, y2_ndc)
            gl.glEnd()

            # Draw the outline
            gl.glColor3f(1.0, 1.0, 1.0)  # Set outline color to white
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex2f(x1_ndc, y1_ndc)
            gl.glVertex2f(x2_ndc, y1_ndc)
            gl.glVertex2f(x2_ndc, y2_ndc)
            gl.glVertex2f(x1_ndc, y2_ndc)
            gl.glEnd()

            # Restore OpenGL state
            gl.glEnable(gl.GL_TEXTURE_2D)  # Re-enable texturing

