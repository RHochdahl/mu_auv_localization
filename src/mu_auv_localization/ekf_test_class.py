
import numpy as np
import rospy

from meas_model_class import MeasurementModelDistances
from process_model_class import ProcessModel


class ExtendedKalmanFilter(object):
    def __init__(self, dim_state, dim_meas, measurement_model, process_model, x0, p0_mat):
        self.dim_state = dim_state
        self.dim_meas = dim_meas
        self._x_est_0 = x0
        self._x_est = self._x_est_0
        self._x_est_last = self._x_est
        self._last_time_stamp_update = rospy.get_time()
        self._last_time_stamp_prediction = rospy.get_time()
        self._p0_mat = p0_mat
        self._p_mat = self._p0_mat

        self.process_model = process_model
        self.measurement_model = measurement_model

    def get_x_est(self):
        return self._x_est

    def get_p_mat(self):
        return self._p_mat

    def reset(self, x_est_0=None, p0_mat=None):
        if x_est_0:
            self._x_est = x_est_0
        else:
            self._x_est = self._x_est_0
        if p0_mat:
            self._p_mat = p0_mat
        else:
            self._p_mat = self._p0_mat

    def predict(self, dt):

        self._x_est = self.process_model.f(self._x_est, dt)
        a_mat = self.process_model.f_jacobian(self._x_est, dt)
        self._p_mat = np.matmul(np.matmul(a_mat, self._p_mat), a_mat.transpose()) + self.process_model.V

        return True

    def _update(self, y, h_mat, w_mat):
        """ helper function for general update """

        # compute K gain
        tmp = np.matmul(np.matmul(h_mat, self._p_mat), h_mat.transpose()) + w_mat
        k_mat = np.matmul(np.matmul(self._p_mat, h_mat.transpose()), np.linalg.inv(tmp))

        # update state
        self._x_est = self._x_est + np.matmul(k_mat, y)

        # update covariance
        p = np.eye(self.dim_state) - np.matmul(k_mat, h_mat)
        self._p_mat = np.matmul(p, self._p_mat)

        return True

    def update_vision_data(self, measurements, detected_tags):

        z_est_vision = self.measurement_model.h_vision_data(self._x_est, detected_tags)
        h_mat_vision = self.measurement_model.h_jacobian_vision_data(self._x_est, detected_tags)
        w_mat_vision_dyn = self.measurement_model.vision_dynamic_meas_model(self._x_est, measurements, detected_tags)

        y = measurements - z_est_vision
        self._update(y, h_mat_vision, w_mat_vision_dyn)

        return True

    def update_orientation_data(self, measurements):
        # measurement is: roll, pitch from /mavros/local_position/pose

        z_est_orient = self.measurement_model.h_orientation_data(self._x_est)
        h_mat_orient = self.measurement_model.h_jacobian_orientation_data()

        y_orientation = measurements - z_est_orient
        self._update(y_orientation, h_mat_orient, self.measurement_model.w_mat_orientation)

        return True

    def update_imu_data(self, measurements, w_mat_imu):
        # measurement is either: body rates + lin. acceleration
        #               or: only body rates

        # currently only using body rates
        z_est_imu = self.measurement_model.h_imu_data(self._x_est)
        h_mat_imu = self.measurement_model.h_jacobian_imu_data()
        y_imu = measurements - z_est_imu
        self._update(y_imu, h_mat_imu, w_mat_imu)

        return True

    @staticmethod
    def roll_pitch_yaw_to_quat(roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr
        qw = cy * cp * cr + sy * sp * sr
        return qx, qy, qz, qw

