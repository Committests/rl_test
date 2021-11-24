# coding=utf-8
import math
from ctypes import byref
from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np
import pyglet
import pyglet.image
from PIL import Image
from pyglet import gl
from pyglet.gl import GLubyte

from duckietown_world import get_texture_file
from . import logger


def get_texture(tex_name: str, rng=None, segment: bool = False) -> "Texture":
    paths = get_texture_file(tex_name)

    if rng:
        path_idx = rng.randint(0, len(paths))
        path = paths[path_idx]
    else:
        path = paths[0]

    oldpath = path
    if segment:
        path += ".SEGMENTED"

    if path not in Texture.tex_cache:
        Texture.tex_cache[path] = Texture(load_texture(oldpath, segment), tex_name=tex_name, rng=rng)

    return Texture.tex_cache[path]


class Texture:
    """
    Manage the caching of textures, and texture randomization
    """

    # Cache of textures
    tex_cache = {}

    def __init__(self, tex, tex_name, rng):
        assert not isinstance(tex, str)
        self.tex = tex
        self.tex_name = tex_name
        self.rng = rng

    def bind(self, segment=False):
        if segment:
            self = get_texture(self.tex_name, self.rng, True)

        gl.glBindTexture(self.tex.target, self.tex.id)


def should_segment_out(tex_path):
    for yes in ["sign", "trafficlight", "asphalt"]:
        if yes in tex_path:
            return True
    for no in ["left", "right", "way", "curve", "straight"]:
        if no in tex_path:
            return False
    return True


@lru_cache(maxsize=None)
def load_texture(tex_path: str, segment: bool = False, segment_into_color=None):
    """segment_into_black controls what type of segmentation we apply: for tiles and all ground textures,
    replacing
    unimportant stuff with black is a good idea. For other things, replacing it with transparency is good too
    (for example, we don't want black traffic lights, because they go over the roads, and they'd cut our
    view of
    things).
    """
    if segment_into_color is None:
        segment_into_color = [0, 0, 0]
    logger.debug(f"loading texture: {tex_path}")
    img = pyglet.image.load(tex_path)
    # img_format = 'RGBA'
    # pitch = img.width * len(img_format)
    # pixels = img.get_data(img_format, pitch)
    #
    #
    # for i in range(x, width):
    #     for j in range(y, height):
    #         pixels[i, j] = (0, 0, 0, 0)

    if segment:
        if should_segment_out(tex_path):  # replace all by 'segment_into_color'
            # https://gamedev.stackexchange.com/questions/55945/how-to-draw-image-in-memory-manually-in-pyglet
            to_fill = np.ones((img.height, img.width), dtype=int)
            to_fill = np.kron(to_fill, np.array(segment_into_color, dtype=int))
            to_fill = list(to_fill.flatten())
            rawData = (GLubyte * len(to_fill))(*to_fill)
            img = pyglet.image.ImageData(img.width, img.height, "RGB", rawData)
        else:  # replace asphalt by black
            # https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55

            im = cv2.imread(tex_path, cv2.IMREAD_UNCHANGED)

            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            # Threshold of blue in HSV space

            lower = np.array([0, 0, 0], dtype="uint8")
            upper = np.array([179, 100, 160], dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_not(mask)

            kernel1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.uint8)
            kernel2 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
            hitormiss1 = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel1)
            hitormiss2 = cv2.morphologyEx(hitormiss1, cv2.MORPH_ERODE, kernel2)
            mask = cv2.bitwise_and(hitormiss1, hitormiss2)

            result = cv2.bitwise_and(hsv, hsv, mask=mask)
            im = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

            rows, cols, channels = im.shape

            raw_img = Image.fromarray(im).tobytes()

            top_to_bottom_flag = -1
            bytes_per_row = channels * cols
            img = pyglet.image.ImageData(
                width=cols, height=rows, format="BGR", data=raw_img, pitch=top_to_bottom_flag * bytes_per_row
            )

    tex = img.get_texture()
    # if img.width == img.height:
    #     tex = tex.get_mipmapped_texture()
    gl.glEnable(tex.target)
    gl.glBindTexture(tex.target, tex.id)
    rawimage = img.get_image_data()

    if tex_path.endswith("jpg"):
        image_data = rawimage.get_data("RGB", img.width * 3)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            img.width,
            img.height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            image_data,
        )

    else:
        image_data = rawimage.get_data("RGBA", img.width * 4)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            img.width,
            img.height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            image_data,
        )

    return tex


def create_frame_buffers(width: int, height: int, num_samples: int) -> Tuple[int, int]:
    """Create the frame buffer objects"""

    # Create a frame buffer (rendering target)
    multi_fbo = gl.GLuint(0)
    gl.glGenFramebuffers(1, byref(multi_fbo))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, multi_fbo)

    # The try block here is because some OpenGL drivers
    # (Intel GPU drivers on macbooks in particular) do not
    # support multisampling on frame buffer objects
    # noinspection PyBroadException
    try:
        # Create a multisampled texture to render into
        fbTex = gl.GLuint(0)
        gl.glGenTextures(1, byref(fbTex))
        gl.glBindTexture(gl.GL_TEXTURE_2D_MULTISAMPLE, fbTex)
        gl.glTexImage2DMultisample(
            gl.GL_TEXTURE_2D_MULTISAMPLE, num_samples, gl.GL_RGBA32F, width, height, True
        )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D_MULTISAMPLE, fbTex, 0
        )

        # Attach a multisampled depth buffer to the FBO
        depth_rb = gl.GLuint(0)
        gl.glGenRenderbuffers(1, byref(depth_rb))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rb)
        gl.glRenderbufferStorageMultisample(
            gl.GL_RENDERBUFFER, num_samples, gl.GL_DEPTH_COMPONENT, width, height
        )
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rb)

    except BaseException as e:
        # logger.warning(e=traceback.format_exc())
        logger.debug("Falling back to non-multisampled frame buffer")

        # Create a plain texture texture to render into
        fbTex = gl.GLuint(0)
        gl.glGenTextures(1, byref(fbTex))
        gl.glBindTexture(gl.GL_TEXTURE_2D, fbTex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, fbTex, 0)

        # Attach depth buffer to FBO
        depth_rb = gl.GLuint(0)
        gl.glGenRenderbuffers(1, byref(depth_rb))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rb)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, width, height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rb)

    # Sanity check

    if pyglet.options["debug_gl"]:
        res = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        assert res == gl.GL_FRAMEBUFFER_COMPLETE

    # Create the frame buffer used to resolve the final render
    final_fbo = gl.GLuint(0)
    gl.glGenFramebuffers(1, byref(final_fbo))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, final_fbo)

    # Create the texture used to resolve the final render
    fbTex = gl.GLuint(0)
    gl.glGenTextures(1, byref(fbTex))
    gl.glBindTexture(gl.GL_TEXTURE_2D, fbTex)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, fbTex, 0)

    if pyglet.options["debug_gl"]:
        res = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        assert res == gl.GL_FRAMEBUFFER_COMPLETE

    # Enable depth testing
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Unbind the frame buffer
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    return multi_fbo, final_fbo


def rotate_point(px, py, cx, cy, theta):
    """
    Rotate a 2D point around a center
    """

    dx = px - cx
    dy = py - cy

    new_dx = dx * math.cos(theta) + dy * math.sin(theta)
    new_dy = dy * math.cos(theta) - dx * math.sin(theta)

    return cx + new_dx, cy + new_dy


def gen_rot_matrix(axis0: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotation matrix for a counterclockwise rotation around the given axis
    """

    axis = axis0 / math.sqrt(np.dot(axis0, axis0))
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)

    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )


def bezier_point(cps, t):
    """
    Cubic Bezier curve interpolation
    B(t) = (1-t)^3 * P0 + 3t(1-t)^2 * P1 + 3t^2(1-t) * P2 + t^3 * P3
    """

    p = ((1 - t) ** 3) * cps[0, :]
    p += 3 * t * ((1 - t) ** 2) * cps[1, :]
    p += 3 * (t ** 2) * (1 - t) * cps[2, :]
    p += (t ** 3) * cps[3, :]

    return p


def bezier_tangent(cps, t):
    """
    Tangent of a cubic Bezier curve (first order derivative)
    B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)
    """

    p = 3 * ((1 - t) ** 2) * (cps[1, :] - cps[0, :])
    p += 6 * (1 - t) * t * (cps[2, :] - cps[1, :])
    p += 3 * (t ** 2) * (cps[3, :] - cps[2, :])

    norm = np.linalg.norm(p)
    p /= norm

    return p


def bezier_closest(cps, p, t_bot=0, t_top=1, n=8):
    mid = (t_bot + t_top) * 0.5

    if n == 0:
        return mid

    p_bot = bezier_point(cps, t_bot)
    p_top = bezier_point(cps, t_top)

    d_bot = np.linalg.norm(p_bot - p)
    d_top = np.linalg.norm(p_top - p)

    if d_bot < d_top:
        # noinspection PyTypeChecker
        return bezier_closest(cps, p, t_bot, mid, n - 1)

    # noinspection PyTypeChecker
    return bezier_closest(cps, p, mid, t_top, n - 1)


def bezier_draw(cps, n=20, red=False):
    pts = [bezier_point(cps, i / (n - 1)) for i in range(0, n)]
    gl.glBegin(gl.GL_LINE_STRIP)

    if red:
        gl.glColor3f(1, 0, 0)
    else:
        gl.glColor3f(0, 0, 1)

    for i, p in enumerate(pts):
        gl.glVertex3f(*p)

    gl.glEnd()
    gl.glColor3f(1, 1, 1)

############################################################################################################
############################################################################################################
# @riza

def bezier_draw_points_curve(cps, n=10, red=False):
    from pyglet import gl
    pts = [bezier_point(cps, i/(n-1)) for i in range(0,n)]
    gl.glPointSize(5)
    gl.glBegin(gl.GL_POINTS)

    if red:
        gl.glColor3f(0, 0, 1)
    else:
        gl.glColor3f(1, 0, 0)

    for i, p in enumerate(pts):
        gl.glVertex3f(p[0], 0.01, p[2])

    gl.glEnd()
    gl.glColor3f(1,1,1)


def bezier_draw_line(cps, red=False):
    """
    Draw a line
    :param red:
    :param cps: (x,y,z) coordinates of start&end points of the line
    :return: Draw line
    """
    from pyglet import gl
    gl.glBegin(gl.GL_LINES)
    if red:
        gl.glColor3f(1, 0, 0)
    else:
        gl.glColor3f(0, 1, 0)
    gl.glVertex3f(*cps[0])
    gl.glVertex3f(*cps[1])
    gl.glEnd()


def draw_point(point):
    """
    Draw a point
    :param point: Coordinates of the point
    """
    from pyglet import gl
    gl.glPointSize(7)
    gl.glBegin(gl.GL_POINTS)
    gl.glColor3f(0, 0, 1)
    gl.glVertex3f(point[0], 0.1, point[2])
    gl.glEnd()


def get_linear_bezier(cps, t):
    """
    Linear Bezier curve interpolation
    B(t) = (1-t)*P0 + t*P1
    """
    p = (1-t) * cps[0, :]
    p += t * cps[1, :]
    return p


def rotate_translate(dir_vec, points, new_center):
    """
    Transformation and Rotation between two 2D coordinate frames
    :param new_center: new origin's coordinate in old frame
    :param dir_vec: directory vector
    :param points: The points to be transformed
    :return: The transformed points
    """
    import math
    # Get dir_vec to calculate angle
    x_dir, _, y_dir = dir_vec
    # Get the angle in (-180, 180)
    x1, y1 = 0, 1
    x2, y2 = x_dir, y_dir
    dot = x1 * x2 + y1 * y2  # dot product
    det = x1 * y2 - y1 * x2  # determinant
    angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    theta = -np.rad2deg(angle)
    sin_theta = np.sin(np.deg2rad(theta))
    cos_theta = np.cos(np.deg2rad(theta))
    # Rotation Matrix
    x_ = cos_theta * new_center[0] - sin_theta * new_center[2]
    y_ = sin_theta * new_center[0] + cos_theta * new_center[2]
    x_trans = -x_
    y_trans = -y_

    new_points = np.zeros((len(points), 3))
    for idx, i in enumerate(points):
        # Rotation and Translation
        new_points[idx][0] = cos_theta * points[idx][0] - sin_theta * points[idx][2] + x_trans
        new_points[idx][2] = sin_theta * points[idx][0] + cos_theta * points[idx][2] + y_trans

    return new_points


def compute_dist(cps, points, dir_vec, n=12, debug=False, red=False):

    # Sample points from dir_line
    pts = [get_linear_bezier(cps, t) for t in np.linspace(0, 1, n)]
    # (y2-y1)/(x2-x1) -> slope of dir_line
    slope = (cps[0][2] - cps[1][2]) / (cps[0][0] - cps[1][0])
    # store features
    features = np.zeros((len(pts), 2))
    DIST_NOT_INTERSECT = 0
    # For each point, draw the perpendicular line
    for i, p in enumerate(pts):

        # Transform points
        transformed_points = rotate_translate(dir_vec, points[:50], p)
        # Find 2 closest points to y=0
        y = transformed_points[:, 2]
        y_pos = y[y > 0]
        y_neg = y[y < 0]

        # This means there's no intersection
        if len(y_pos) == 0 or len(y_neg) == 0:
            # For top 3 points, look for points in next_tile
            if i < (n/2):
                # Transform points
                transformed_points = rotate_translate(dir_vec, points[50:100], p)
            # For below 3 points, look for points in prev_tile
            else:
                # Transform points
                transformed_points = rotate_translate(dir_vec, points[100:], p)

            # Find 2 closest points to y=0
            y = transformed_points[:, 2]
            y_pos = y[y > 0]
            y_neg = y[y < 0]
            if len(y_pos) == 0 or len(y_neg) == 0:

                # TODO: ************ CHANGE THIS PART ************
                # For top 3 points, look for points in next_tile
                if i > (n/2):
                    # Transform points
                    transformed_points = rotate_translate(dir_vec, points[50:100], p)
                # For below 3 points, look for points in prev_tile
                else:
                    # Transform points
                    transformed_points = rotate_translate(dir_vec, points[100:], p)
                # Find 2 closest points to y=0
                y = transformed_points[:, 2]
                y_pos = y[y > 0]
                y_neg = y[y < 0]
                if len(y_pos) == 0 or len(y_neg) == 0:
                # TODO: ************ CHANGE THIS PART ************


                    features[i] = [0, DIST_NOT_INTERSECT]
                    continue

        p1_idx = np.where(y == np.max(y_neg))[0][0]
        p2_idx = np.where(y == np.min(y_pos))[0][0]

        p1 = transformed_points[p1_idx]
        p2 = transformed_points[p2_idx]

        # Find intersection between that line & y=0
        # Get the slope of the line that connects p1 & p2
        slope_2 = (p1[2] - p2[2]) / (p1[0] - p2[0])
        # Get the line equation
        k2 = p1[2] - slope_2 * p1[0]
        # Get the intersection
        x_intersection = -k2 / slope_2
        # Compute distance between intersection point to origin
        dist = np.linalg.norm(x_intersection)
        # Check where intersection is, right or left side of center line
        # Left --> -dist,   Right --> dist
        if x_intersection < 0:
            dist *= -1

        # Draw line from origin to x_intersect
        # Origin in old-frame
        start = p
        # Inverse rotate x_intersect to get old-frame coords.
        end = inverse_rotate_translate(dir_vec, [x_intersection, 0, 0], p)
        # Draw line from origin the x_intersect
        bezier_draw_line(np.vstack((start, end)), red=red)

        # Draw it (DEBUG)
        if debug:
            # Draw intersection point
            # from pyglet import gl
            # gl.glPointSize(4)
            # gl.glBegin(gl.GL_POINTS)
            # gl.glColor3f(1, 1, 1)
            # gl.glVertex3f(end[0], 0.1, end[2])
            # gl.glEnd()

            # Draw x-axis
            # k = y + x/slope --> equation of each sensing(perp.) line
            k = p[2] + p[0] / slope
            # Sample one point from that line to find the unit vector
            p_1 = np.array([0, 0.01, k])
            # get unit dir_vec of perpendicular line
            x_, _, y_ = (p_1 - p)
            norm = np.linalg.norm([x_, y_])
            x_ /= norm
            y_ /= norm
            # Get 2 points from uni dir_vec for drawing
            dir_start = [p[0] + 0.2 * x_, 0.01, p[2] + 0.2 * y_]
            dir_end = [p[0] - 0.2 * x_, 0.01, p[2] - 0.2 * y_]
            # bezier_draw_line(np.vstack((dir_start, dir_end)))

        features[i] = [1, dist]
    return features


def inverse_rotate_translate(dir_vec, point, new_center):
    """
    Inverse Transformation and Rotation between two 2D coordinate frames
    :param new_center: new origin's coordinate in old frame
    :param dir_vec: directory vector
    :param point: The point to be inverse transformed
    :return: The transformed point
    """
    import math
    x_dir, _, y_dir = dir_vec

    x1, y1 = 0, 1
    x2, y2 = x_dir, y_dir
    dot = x1 * x2 + y1 * y2  # dot product
    det = x1 * y2 - y1 * x2  # determinant
    angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    theta = -np.rad2deg(angle)

    sin_theta = np.sin(np.deg2rad(theta))
    cos_theta = np.cos(np.deg2rad(theta))

    x_ = cos_theta * new_center[0] - sin_theta * new_center[2]
    y_ = sin_theta * new_center[0] + cos_theta * new_center[2]
    x_trans = -x_
    y_trans = -y_

    new_point = [0, 0, 0]
    new_point[2] = cos_theta * (point[2]-y_trans) - sin_theta * (point[0] - x_trans)
    new_point[0] = (point[0] - x_trans + sin_theta * new_point[2]) / cos_theta

    return new_point
