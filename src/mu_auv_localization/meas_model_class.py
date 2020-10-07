import numpy as np
from pyquaternion import Quaternion
import rospy


class MeasurementModelDistances(object):
    def __init__(self, dim_state, dim_meas, W):
        self._dim_state = dim_state
        self._dim_meas = dim_meas
        self._W = W

    def h(self):
        pass

    # Jacobian of the measurement function
    def h_jacobian(self, x, tags):
        """

        :param x: current (predicted) state
        :param tags: currently detected tags
        :return: H,
        """

        num_tags = tags.shape[0]
        H = np.zeros((num_tags * self._dim_meas, self._dim_state))

        for i, tag in enumerate(tags):
            tag_pos = tag
            # dist = sqrt((x - x_tag) ^ 2 + (y - y_tag) ^ 2 + (z - z_tag) ^ 2)
            dist = np.sqrt((x[0] - tag_pos[0]) ** 2 +
                           (x[1] - tag_pos[1]) ** 2 +
                           (x[2] - tag_pos[2]) ** 2)

            # dh /dx1 = 1/2 * (dist ^2)^(-1/2) * (2 * (x1 - t1) * 1)
            h_jac_x1 = 0.5 * 2.0 * (x[0] - tag_pos[0]) / dist
            # dh /dx2
            h_jac_x2 = 0.5 * 2.0 * (x[1] - tag_pos[1]) / dist
            # dh /dx3
            h_jac_x3 = 0.5 * 2.0 * (x[2] - tag_pos[2]) / dist

            H[self._dim_meas * i, :] = [h_jac_x1, h_jac_x2, h_jac_x3, 0, 0]
            H[self._dim_meas * i + 1, :] = [0, 0, 0, 1.0, 0]

        return H  # dim [num_tag*num_meas X num_states]

    def dynamic_meas_model(self, x, z):

        num_tags = z.shape[0]
        # initialize dynamic W
        W_tilde = np.zeros((num_tags * self._dim_meas, num_tags * self._dim_meas))

        for i, tag in enumerate(z):
            tag_pos = tag[-1-4:-1]
            # dist = sqrt((x - x_tag) ^ 2 + (y - y_tag) ^ 2 + (z - z_tag) ^ 2)
            dist = np.sqrt((x[0] - tag_pos[0]) ** 2 +
                           (x[1] - tag_pos[1]) ** 2 +
                           (x[2] - tag_pos[2]) ** 2)
            # add dynamic noise to measurement noise for distance measurement
            penalty_dist = 5  # todo
            penalty_yaw = 5  # todo
            W_tilde[self._dim_meas * i, self._dim_meas * i] = dist / ((x[2] - tag_pos[2]) * penalty_dist) + self._W[0, 0]
            # add dynamic noise to measurement noise for yaw measurement
            W_tilde[self._dim_meas * i + 1, self._dim_meas * i + 1] = dist / ((x[2] - tag_pos[2]) * penalty_yaw) + self._W[1, 1]

        return W_tilde
