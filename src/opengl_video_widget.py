import cupy as cp
import cv2
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import OpenGL.GL as gl
import freetype


class OpenGLVideoWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.texture_id = None
        self.pbo_id = None
        self.image = None
        self.bounding_boxes = []
        self.confidences = []
        self.classes = []

        # Initialize FreeType and load font
        self.face = freetype.Face("resources/fonts/consolas.ttf")  # Replace with your font path
        self.face.set_char_size(34 * 64)  # Adjust font size if necessary

    def initializeGL(self):
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0, 0, 0, 1)
        gl.glEnable(gl.GL_TEXTURE_2D)
        self.init_pbo()

    def resizeEvent(self, event):
        window_width = self.parent().width()
        window_height = self.parent().height()
        aspect_ratio = 16 / 9

        if window_width / window_height > aspect_ratio:
            new_width = int(window_height * aspect_ratio)
            new_height = window_height
        else:
            new_width = window_width
            new_height = int(window_width / aspect_ratio)

        x_offset = (window_width - new_width) // 2
        y_offset = (window_height - new_height) // 2

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

    def upload_frame_to_opengl(self, frame, bounding_boxes, confidences, classes):
        self.image = frame
        self.bounding_boxes = bounding_boxes
        self.confidences = confidences
        self.classes = classes
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cp.flipud(cp.array(frame_rgb))
        frame_rgb_np = cp.asnumpy(frame_rgb)

        if self.texture_id is None:
            self.texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, frame_rgb_np.shape[1], frame_rgb_np.shape[0], 0, gl.GL_RGB,
                            gl.GL_UNSIGNED_BYTE, frame_rgb_np)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        else:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, frame_rgb_np.shape[1], frame_rgb_np.shape[0], gl.GL_RGB,
                               gl.GL_UNSIGNED_BYTE, frame_rgb_np)

        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self.update()

    def draw_texture(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0);
        gl.glVertex3f(-1, -1, 0)
        gl.glTexCoord2f(1, 0);
        gl.glVertex3f(1, -1, 0)
        gl.glTexCoord2f(1, 1);
        gl.glVertex3f(1, 1, 0)
        gl.glTexCoord2f(0, 1);
        gl.glVertex3f(-1, 1, 0)
        gl.glEnd()
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def draw_bounding_boxes(self):
        gl.glLineWidth(2)
        for i in range(len(self.bounding_boxes)):
            gl.glColor3f(*self.classes[i])  # Set color based on class
            bbox = self.bounding_boxes[i]
            confidence = self.confidences[i]
            cls = self.classes[i]
            x1 = (bbox[0] / self.image.shape[1]) * 2 - 1
            y1 = 1 - (bbox[1] / self.image.shape[0]) * 2
            x2 = (bbox[2] / self.image.shape[1]) * 2 - 1
            y2 = 1 - (bbox[3] / self.image.shape[0]) * 2

            # Draw the bounding box
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex2f(x1, y1)
            gl.glVertex2f(x2, y1)
            gl.glVertex2f(x2, y2)
            gl.glVertex2f(x1, y2)
            gl.glEnd()

            # Render and draw the confidence score text
            text = f"{confidence:.2f}"
            text_texture_id, text_width, text_height = self.render_text_to_texture(text)

            # Calculate text position based on bounding box
            text_x = x1  # Small padding to the left of the box
            text_y = y1 + 0.02  # Small padding above the box

            # Draw the text texture
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, text_texture_id)

            gl.glBegin(gl.GL_QUADS)
            gl.glTexCoord2f(0, 1);
            gl.glVertex2f(text_x, text_y)
            gl.glTexCoord2f(1, 1);
            gl.glVertex2f(text_x + (text_width / self.image.shape[1]), text_y)
            gl.glTexCoord2f(1, 0);
            gl.glVertex2f(text_x + (text_width / self.image.shape[1]), text_y + (text_height / self.image.shape[0]))
            gl.glTexCoord2f(0, 0);
            gl.glVertex2f(text_x, text_y + (text_height / self.image.shape[0]))
            gl.glEnd()

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glDeleteTextures([text_texture_id])  # Clean up texture after drawing

        gl.glColor3f(1, 1, 1)  # Reset color

    def render_text_to_texture(self, text):
        # Set pixel storage alignment to handle textures with non-multiple-of-4 widths
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        # Initialize dimensions for the final texture
        total_width = 0
        max_height = 0
        char_bitmaps = []

        # Loop through each character in the text to get individual glyph bitmaps
        for char in text:
            self.face.load_char(char)
            bitmap = self.face.glyph.bitmap
            char_bitmaps.append(
                (bitmap.buffer, bitmap.width, bitmap.rows, self.face.glyph.bitmap_left, self.face.glyph.bitmap_top))
            total_width += bitmap.width  # Accumulate width for each character
            max_height = max(max_height, bitmap.rows)  # Track max height for texture height

        # Create a combined buffer for the text with correct width and height
        text_data = bytearray(total_width * max_height)

        # Copy each character's bitmap into the combined buffer with proper alignment
        x_offset = 0
        for buffer, width, height, bitmap_left, bitmap_top in char_bitmaps:
            y_offset = max_height - bitmap_top  # Adjust based on the glyph's vertical offset from baseline
            for y in range(height):
                for x in range(width):
                    text_data[((y_offset + y) * total_width) + x_offset + x] = buffer[y * width + x]
            x_offset += width

        # Generate OpenGL texture from the combined buffer
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_ALPHA, total_width, max_height, 0, gl.GL_ALPHA, gl.GL_UNSIGNED_BYTE,
                        text_data)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        return texture_id, total_width, max_height
