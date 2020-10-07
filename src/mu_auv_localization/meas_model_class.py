import numpy as np


class MeasurementModelDistances(object):
    def __init__(self, dim_state, dim_meas, W, c_penalty_dist, c_penalty_yaw):
        self._dim_state = dim_state
        self._dim_meas = dim_meas
        self._W = W
        self._c_penalty_dist = c_penalty_dist
        self._c_penalty_yaw = c_penalty_yaw

    def h(self, x_est, detected_tags):

        num_tags = detected_tags.shape[0]
        z_est = np.zeros((num_tags * self._dim_meas, 1))

        for i, tag in enumerate(detected_tags):
            tag_pos = tag[1:4]
            dist = self.get_dist(x_est, tag_pos)
            yaw = x_est[5]

            z_est[i * self._dim_meas, 0] = dist
            z_est[i * self._dim_meas + 1, 0] = yaw

        return z_est  # dim [num_tags*dim_meas X 1]

    # Jacobian of the measurement function
    def h_jacobian(self, x_est, detected_tags):

        num_tags = detected_tags.shape[0]
        H = np.zeros((num_tags * self._dim_meas, self._dim_state))

        for i, tag in enumerate(detected_tags):
            tag_pos = tag[1:4]
            dist = self.get_dist(x_est, tag_pos)

            # dh /dx = 1/2 * (dist ^2)^(-1/2) * (2 * (x1 - t1) * 1)
            h_jac_x = 0.5 * 2.0 * (x_est[0] - tag_pos[0]) / dist
            # dh /dy
            h_jac_y = 0.5 * 2.0 * (x_est[1] - tag_pos[1]) / dist
            # dh /dz
            h_jac_z = 0.5 * 2.0 * (x_est[2] - tag_pos[2]) / dist
            # dh /dyaw
            h_jac_yaw = 1.0

            H[self._dim_meas * i, 0:3] = [h_jac_x, h_jac_y, h_jac_z]
            H[self._dim_meas * i + 1, 3] = h_jac_yaw
            # all other derivatives are zero

        return H  # dim [num_tags*dim_meas X dim_state]

    def dynamic_meas_model(self, x_est, measurements, detected_tags):
        # currently not using measured tag-position in camera coordinates, but known tag position from calibration,
        # since currently not possible to (nicely) pass full tag-pose measurement to method

        num_tags = detected_tags.shape[0]
        # initialize dynamic W
        W_dyn = np.zeros((num_tags * self._dim_meas, num_tags * self._dim_meas))

        for i, tag in enumerate(detected_tags):
            tag_pos = tag[1:4]
            # dist = sqrt((x - x_tag) ^ 2 + (y - y_tag) ^ 2 + (z - z_tag) ^ 2)
            dist = self.get_dist(x_est, tag_pos)
            # add dynamic noise to measurement noise for distance measurement
            W_dyn[self._dim_meas * i, self._dim_meas * i] = dist / ((x_est[2] - tag_pos[2])
                                                                    * self._c_penalty_dist) + self._W[0, 0]
            # add dynamic noise to measurement noise for yaw measurement
            W_dyn[self._dim_meas * i + 1, self._dim_meas * i + 1] = dist / ((x_est[2] - tag_pos[2])
                                                                            * self._c_penalty_yaw) + self._W[1, 1]

        return W_dyn  # dim [num_tags*dim_meas X num_tag*dim_meas]

    def get_dist(self, x_est, tag_pos):
        # dist = sqrt((x - x_tag) ^ 2 + (y - y_tag) ^ 2 + (z - z_tag) ^ 2)
        dist = np.sqrt((x_est[0] - tag_pos[0]) ** 2 +
                       (x_est[1] - tag_pos[1]) ** 2 +
                       (x_est[2] - tag_pos[2]) ** 2)
        return dist
