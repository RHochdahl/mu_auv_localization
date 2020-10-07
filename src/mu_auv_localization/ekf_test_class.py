
import numpy as np
from pyquaternion import Quaternion
import rospy

from meas_model_class import MeasurementModelDistances
from process_model_class import ProcessModel


class ExtendedKalmanFilter(object):
    def __init__(self, dim_state, dim_meas, measurement_model, process_model, x0, P0):
        self.dim_state = dim_state
        self.dim_meas = dim_meas
        self._x_est_0 = x0
        self._x_est = self._x_est_0
        self._x_est_last = self._x_est
        self._last_time_stamp_update = rospy.get_time()
        self._last_time_stamp_prediction = rospy.get_time()
        # covariance of x
        self._P0 = P0
        self._P = self._P0

        self.process_model = process_model
        self.measurement_model = measurement_model

    def yaw_pitch_roll_to_quat(self, yaw, pitch, roll):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        return (Quaternion(x=cy * cp * sr - sy * sp * cr, y=sy * cp * sr + cy * sp * cr, z=sy * cp * cr - cy * sp * sr,
                           w=cy * cp * cr + sy * sp * sr))

    def set_x_0(self, x0):
        self._x_est = x0
        return True

    def set_P_0(self, P_0):
        self._P = P_0
        return True

    def reset(self):
        self._x_est = self._x_est_0
        self._P = self._P0

    def predict(self, dt):

        self._x_est = self.process_model.f(self._x_est, dt)
        A = self.process_model.f_jacobian(self._x_est, dt)
        P = np.matmul(A, self._P)
        self._P = np.matmul(P, A.transpose()) + self.process_model.V

        return True

    def update(self, measurements, detected_tags):

        z_est = self.measurement_model.h(detected_tags)
        H = self.measurement_model.h_jacobian(measurements, detected_tags)
        W_dyn = self.measurement_model(measurements, detected_tags)

        y = measurements - z_est
        # compute K gain
        K = np.matmul(self._P, H.transpose())
        tmp = np.matmul(H, self._P)
        tmp = np.matmul(tmp, H.transpose()) + W_dyn
        K = np.matmul(K, np.linalg.inv(tmp))

        # update state
        self._x_est = self._x_est + np.matmul(K, y)

        # update covariance
        P = np.eye(self.dim_state) - np.matmul(K, H)
        self._P = np.matmul(P, self._P)

        return True


