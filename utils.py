import numpy as np
import cv2

class CameraInfo:
    """
    Class to represent the camera information.

    :param width: Camera width in pixels.
    :param height: Camera height in pixels.
    :param camera_matrix: Intrinsic camera matrix.
    :param distortion_coeffs: Distortion coefficients.
    """

    def __init__(self, width, height, camera_matrix, distortion_coeffs):
        self.width = width
        self.height = height
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs

class Pose:
    """
    Class to represent a pose in SE(3).

    :param t: float, timestamp of the pose.
    :param position: 3D position vector.
    :param quaternion: Quaternion representing orientation.
    """

    def __init__(self, t, position, quaternion):
        self.t = t
        self.position = position
        self.quaternion = quaternion
    def __repr__(self):
        return f"Pose(t={self.t}, position={self.position}, quaternion={self.quaternion})"

class Motion:
    """
    Represents the motion between two poses in 6DoF.

    :param vel: Linear velocity (in m/s).
    :param omega: Angular velocity (in rad/s).
    :param t: timestamp of motion in seconds.
    :param dt: Time difference between the two poses (in seconds).
    """

    def __init__(self, vel, omega, t=None, dt=None):
        self.vel = vel
        self.omega = omega
        self.t = t
        self.dt = dt

class FlowData:
    """
    Class to represent flow data, both distorted and undistorted.

    :param dist_x: ndarray, distorted pixel column (x-coordinate).
    :param dist_y: ndarray, distorted pixel row (y-coordinate).
    :param dist_u: ndarray, distorted flow in the x-axis direction.
    :param dist_v: ndarray, distorted flow in the y-axis direction.
    :param undist_x: ndarray, undistorted pixel column (x-coordinate).
    :param undist_y: ndarray, undistorted pixel row (y-coordinate).
    :param undist_u: ndarray, undistorted flow in the x-axis direction.
    :param undist_v: ndarray, undistorted flow in the y-axis direction.
    """

    def __init__(self, dist_x, dist_y, dist_u, dist_v, undist_x=None, undist_y=None, undist_u=None, undist_v=None):
        self.dist_x = dist_x
        self.dist_y = dist_y
        self.dist_u = dist_u
        self.dist_v = dist_v
        self.undist_x = undist_x
        self.undist_y = undist_y
        self.undist_u = undist_u
        self.undist_v = undist_v

    def visualize_flow(self, camera_info: CameraInfo):
        """
        Visualize the flow data using the camera matrix and distortion coefficients.

        :param camera_info: CameraInfo, containing camera_matrix and distortion coefficients.
        :return bgr: np.array, colored flow image.
        """
        # Fetch data.
        i = self.dist_y.astype(np.int32)
        j = self.dist_x.astype(np.int32)
        u = self.undist_u.astype(np.float32)
        v = self.undist_v.astype(np.float32)

        # Compute magnitude and angle of the flow.
        magnitude, angle = cv2.cartToPolar(u, v)

        # Normalize magnitude to the range [0, 255].
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Create an HSV image of size width x height.
        hsv = np.zeros((camera_info.height, camera_info.width, 3), dtype=np.uint8)

        # Set the hue based on the angle and value based on the magnitude.
        hsv[i, j, 0] = (angle * 180 / np.pi / 2).flatten()  # Scale angle to [0, 180].
        hsv[i, j, 1] = 255  # Maximum saturation
        hsv[i, j, 2] = magnitude_normalized.flatten()  # Set value based on normalized magnitude.

        # Convert HSV to BGR for visualization.
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

class EventStream:
    """
    Class to represent a stream of events from an event camera.

    :param t: np.array, (N,), event timestamps in seconds (float32).
    :param x: np.array, (N,), x-coordinates of events (pixel width, int).
    :param y: np.array, (N,), y-coordinates of events (pixel height, int).
    :param polarity: np.array, (N,), polarities of events (int).
    """

    def __init__(self, t, x, y, polarity):
        self.t = t
        self.x = x
        self.y = y
        self.polarity = polarity

    def __len__(self):
        return len(self.t)

    def event_image(self, camera_info: CameraInfo):
        """
        Generates a color image representing the event stream.
        Red represents events with polarity 0.
        Blue represents events with polarity 1.
        Out-of-bounds events are ignored (motion compensated case).

        :return: np.array, (height, width, 3), color image.
        """
        # Create an empty image with 3 channels (for color) initialized to zero.
        img = np.zeros((camera_info.height, camera_info.width, 3), dtype=np.uint8)

        # Create a mask for valid x and y coordinates within the image dimensions.
        valid_mask = (self.x >= 0) & (self.x < camera_info.width) & (self.y >= 0) & (self.y < camera_info.height)

        # Red for polarity 0 (use valid x, y coordinates).
        red_mask = valid_mask & (self.polarity == 0)
        img[self.y[red_mask], self.x[red_mask], :] = [0, 0, 255]  # Red in BGR format.

        # Blue for polarity 1.
        blue_mask = valid_mask & (self.polarity == 1)
        img[self.y[blue_mask], self.x[blue_mask], :] = [255, 0, 0]  # Blue in BGR format.

        return img

    def count_time_image(self, camera_info: CameraInfo):
        """
        Computes the count and mean timestamp values for each pixel.

        :return: Tuple[np.array, np.array], (count image, norm mean time image).
        """
        # Create a single-channel grayscale image initialized to zero.
        count_img = np.zeros((camera_info.height, camera_info.width), dtype=np.uint32)
        time_img  = np.zeros((camera_info.height, camera_info.width), dtype=np.float32)

        # Create a mask for valid x and y coordinates within the image dimensions.
        valid_mask = (self.x >= 0) & (self.x < camera_info.width) & (self.y >= 0) & (self.y < camera_info.height)

        # Increment counts and times at valid positions.
        np.add.at(count_img, (self.y[valid_mask], self.x[valid_mask]), 1)
        np.add.at(time_img, (self.y[valid_mask], self.x[valid_mask]), self.t[valid_mask] - self.t[0])

        # Compute mean timestamps by dividing by count.
        mean_time_img = np.divide(time_img, count_img.astype(np.float32), out=np.zeros_like(time_img), where=count_img != 0)

        return count_img, mean_time_img