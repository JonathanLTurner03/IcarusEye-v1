import cupy as cp
import cv2
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import OpenGL.GL as gl

class OpenGLVideoWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.texture_id = None
        self.pbo_id = None
        self.image = None
        self.bounding_boxes = []
        self.confidences = []

    def initializeGL(self):
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0, 0, 0, 1)
        gl.glEnable(gl.GL_TEXTURE_2D)
        self.init_pbo()

    def resizeEvent(self, event):
        # Get the new size of the widget
        window_width = self.parent().width()
        window_height = self.parent().height()

        # Calculate the aspect ratio
        aspect_ratio = 16 / 9

        # Calculate the new dimensions while maintaining the aspect ratio
        if window_width / window_height > aspect_ratio:
            new_width = int(window_height * aspect_ratio)
            new_height = window_height
        else:
            new_width = window_width
            new_height = int(window_width / aspect_ratio)

        # Center the widget
        x_offset = (window_width - new_width) // 2
        y_offset = (window_height - new_height) // 2

        # Resize and reposition the widget
        self.setGeometry(x_offset, y_offset, new_width, new_height)
        super(OpenGLVideoWidget, self).resizeEvent(event)

    def init_pbo(self):
        self.pbo_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, 1920 * 1080 * 4, None, gl.GL_STREAM_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if self.texture_id is not None:
            self.draw_texture()
        self.draw_bounding_boxes()

    def upload_frame_to_opengl(self, frame, bounding_boxes, confidences):
        self.image = frame
        self.bounding_boxes = bounding_boxes
        self.confidences = confidences
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Correct color conversion
        frame_rgb = cp.flipud(cp.array(frame_rgb))  # Flip vertically to correct the upside-down issue
        frame_rgb_np = cp.asnumpy(frame_rgb)  # Convert CuPy array to NumPy array

        if self.texture_id is None:
            self.texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, frame_rgb_np.shape[1], frame_rgb_np.shape[0], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, frame_rgb_np)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        else:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, frame_rgb_np.shape[1], frame_rgb_np.shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, frame_rgb_np)

        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self.update()

    def draw_texture(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0); gl.glVertex3f(-1, -1, 0)
        gl.glTexCoord2f(1, 0); gl.glVertex3f(1, -1, 0)
        gl.glTexCoord2f(1, 1); gl.glVertex3f(1, 1, 0)
        gl.glTexCoord2f(0, 1); gl.glVertex3f(-1, 1, 0)
        gl.glEnd()
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


    def draw_bounding_boxes(self):
        gl.glColor3f(1, 0, 0)  # Set color to red
        gl.glLineWidth(2)  # Set line width
        for i in range(len(self.bounding_boxes)):
            bbox = self.bounding_boxes[i]
            confidence = self.confidences[i]
            x1 = (bbox[0] / self.image.shape[1]) * 2 - 1
            y1 = 1 - (bbox[1] / self.image.shape[0]) * 2
            x2 = (bbox[2] / self.image.shape[1]) * 2 - 1
            y2 = 1 - (bbox[3] / self.image.shape[0]) * 2
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex2f(x1, y1)
            gl.glVertex2f(x2, y1)
            gl.glVertex2f(x2, y2)
            gl.glVertex2f(x1, y2)
            gl.glEnd()
        gl.glColor3f(1, 1, 1)  # Reset color to white