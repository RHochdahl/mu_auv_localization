
import numpy as np
from pyquaternion import Quaternion
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

    def get_P(self):
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

    def update(self, measurements, detected_tags):

        z_est = self.measurement_model.h(self._x_est, detected_tags)
        h_mat = self.measurement_model.h_jacobian(self._x_est, detected_tags)
        w_mat_dyn = self.measurement_model.dynamic_meas_model(self._x_est, measurements, detected_tags)

        y = measurements - z_est
        # compute K gain
        tmp = np.matmul(np.matmul(h_mat, self._p_mat), h_mat.transpose()) + w_mat_dyn
        k_mat = np.matmul(np.matmul(self._p_mat, h_mat.transpose()), np.linalg.inv(tmp))

        # update state
        self._x_est = self._x_est + np.matmul(k_mat, y)

        # update covariance
        p = np.eye(self.dim_state) - np.matmul(k_mat, h_mat)
        self._p_mat = np.matmul(p, self._p_mat)

        return True

    def update_orientation(self, measurements):

        return True

    def update_angular_vel(self, measurements):

        return True

    # todo: check this, taken from tim
    @staticmethod
    def yaw_pitch_roll_to_quat(yaw, pitch, roll):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        return (Quaternion(x=cy * cp * sr - sy * sp * cr, y=sy * cp * sr + cy * sp * cr, z=sy * cp * cr - cy * sp * sr,
                           w=cy * cp * cr + sy * sp * sr))

