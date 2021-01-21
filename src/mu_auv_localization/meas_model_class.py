from __future__ import print_function
import numpy as np
"""
Different measurement models for the EKF for the visual localization
    - Distances: using distance and yaw angle to tag as measurement
    - ...

EKF state: [x, y, z, roll, pitch, yaw, dx, dy, dz, droll, dpitch, dyaw]
(linear and angular velocities in body frame)
"""


class MeasurementModelDistances(object):
    def __init__(self, dim_state, dim_meas, w_mat_vision, c_penalty_dist,
                 c_penalty_yaw, w_mat_orientation):
        self._dim_state = dim_state
        self._dim_meas = dim_meas
        self._w_mat_vision_static = w_mat_vision
        self._c_penalty_dist = c_penalty_dist
        self._c_penalty_yaw = c_penalty_yaw
        self.w_mat_orientation = w_mat_orientation

    def h_vision_data(self, x_est, detected_tags):
        # measurement is: distance and yaw-angle to each tag
        num_tags = detected_tags.shape[0]
        self.z_est_vision = np.zeros((num_tags * self._dim_meas, 1))
        self.h_mat_vision = np.zeros((num_tags * self._dim_meas, self._dim_state))

        rospy.loginfo("\ndetected_tags: " + str(detected_tags))

        for i, tag in enumerate(detected_tags):
            tag_pos = tag[1:4]
            dist = self.get_dist(x_est, tag_pos)
            yaw = x_est[5]

            self.z_est_vision[i * self._dim_meas, 0] = dist
            self.z_est_vision[i * self._dim_meas + 1, 0] = yaw

            # dh /dx = 1/2 * (dist ^2)^(-1/2) * (2 * (x1 - t1) * 1)
            h_jac_x = (x_est[0] - tag_pos[0]) / dist
            # dh /dy
            h_jac_y = (x_est[1] - tag_pos[1]) / dist
            # dh /dz
            h_jac_z = (x_est[2] - tag_pos[2]) / dist
            # dh /dyaw
            h_jac_yaw = 1.0

            self.h_mat_vision[self._dim_meas * i, 0:3] = [h_jac_x, h_jac_y, h_jac_z]
            self.h_mat_vision[self._dim_meas * i + 1, 5] = h_jac_yaw
            # all other derivatives are zero

        return self.z_est_vision  # dim [num_tags*dim_meas X 1]

    def h_jacobian_vision_data(self):
        return self.h_mat_vision

    def h_orientation_data(self, x_est):
        # measurement is: roll, pitch from /mavros/local_position/pose
        self.z_est_orientation = np.array([x_est[3], x_est[4]]).reshape((-1, 1))

        self.h_mat_orientation = np.zeros((2, self._dim_state))
        # all derivatives zero except for roll, pitch:
        self.h_mat_orientation[0, 3] = 1.0  # dh /droll
        self.h_mat_orientation[1, 4] = 1.0  # dh/ dpitch
        return self.z_est_orientation  # dim: [2 X 1]

    def h_jacobian_orientation_data(self):
        return self.h_mat_orientation

    def h_imu_data(self, x_est, using_lin_acc=False):
        if not using_lin_acc:
            # measurement is: roll rate, pitch rate, yaw rate
            self.z_est_imu = np.array([x_est[9], x_est[10], x_est[11]]).reshape((-1, 1))

            self.h_mat_imu = np.zeros((3, self._dim_state))
            # all derivatives zero except for body rates
            self.h_mat_imu[0, 9] = 1.0
            self.h_mat_imu[1, 10] = 1.0
            self.h_mat_imu[2, 11] = 1.0
        else:
            # measurement is: angular velocities and linear accelerations
            rospy.logfatal(
                "[%s] Using linear acceleration measurements from IMU!" +
                "Not implemented yet", rospy.get_name())
        return self.z_est_imu

    def h_jacobian_imu_data(self):
        return self.h_mat_imu

    def vision_dynamic_meas_model(self, x_est, measurements, detected_tags):
        # currently not using measured tag-position in camera coordinates,
        # but known tag position from calibration, since currently not possible
        # to (nicely) pass full tag-pose measurement to method

        num_tags = detected_tags.shape[0]
        # initialize dynamic W
        w_mat_dyn = np.zeros(
            (num_tags * self._dim_meas, num_tags * self._dim_meas))

        for i, tag in enumerate(detected_tags):
            # tag_pos = tag[1:4]
            # dist = sqrt((x - x_tag) ^ 2 + (y - y_tag) ^ 2 + (z - z_tag) ^ 2)
            # dist = self.get_dist(x_est, tag_pos)

            # add dynamic noise to measurement noise for distance measurement
            # w_mat_dyn[self._dim_meas * i, self._dim_meas * i] = dist / ((x_est[2] - tag_pos[2])
            #                                                  * self._c_penalty_dist) + self._w_mat_vision_static[0, 0]
            # add dynamic noise to measurement noise for yaw measurement
            # w_mat_dyn[self._dim_meas * i + 1, self._dim_meas * i + 1] = dist / ((x_est[2] - tag_pos[2])
            #                                                   * self._c_penalty_yaw) + self._w_mat_vision_static[1, 1]

            # debugging: not dynamic
            # noise for distance measurement
            w_mat_dyn[self._dim_meas * i,
                      self._dim_meas * i] = self._w_mat_vision_static[0, 0]
            # noise for yaw measurement
            w_mat_dyn[self._dim_meas * i + 1,
                      self._dim_meas * i + 1] = self._w_mat_vision_static[1, 1]

        return w_mat_dyn  # dim [num_tags*dim_meas X num_tag*dim_meas]

    def get_dist(self, x_est, tag_pos):
        # dist = sqrt((x - x_tag) ^ 2 + (y - y_tag) ^ 2 + (z - z_tag) ^ 2)
        dist = np.sqrt((x_est[0] - tag_pos[0])**2 +
                       (x_est[1] - tag_pos[1])**2 + (x_est[2] - tag_pos[2])**2)
        return dist
