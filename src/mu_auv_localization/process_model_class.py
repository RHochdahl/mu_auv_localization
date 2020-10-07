import numpy as np
from pyquaternion import Quaternion
import rospy


class ProcessModel(object):
    def __init__(self, dim_state, dim_meas, V):
        self._dim_state = dim_state
        self._dim_meas = dim_meas
        self._V = V

    def f(self, x_last):
        x = x_last

        return x

    def f_jacobian(self):
        A = np.asarray([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])
        return A

