import numpy as np
from pyquaternion import Quaternion
import rospy

# state: [x, y, z, roll, pitch, yaw, dx, dy, dz, droll, dpitch, dyaw]
# linear and angular velocities in body frame
# todo: frame conventions -> work entirely in ROS standards?


class ProcessModel(object):
    def __init__(self, dim_state, dim_meas, V):
        self._dim_state = dim_state
        self._dim_meas = dim_meas
        self.V = V

    def f(self, x_est, dt):
        x = x_est

        return x

    def f_jacobian(self, x_est, dt):
        A = np.eye(self._dim_state)
        return A  # dim [dim_state X dim_state]


