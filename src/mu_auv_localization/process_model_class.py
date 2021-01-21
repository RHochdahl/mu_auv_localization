from __future__ import print_function
import numpy as np
"""
Different process models for the EKF for the visual localization
    - simple model: no prediction, new state = old state
    - ...

EKF state: [x, y, z, roll, pitch, yaw, dx, dy, dz, droll, dpitch, dyaw]
(linear and angular velocities in body frame)
"""


class ProcessModel(object):
    """ Simple process model: no prediction """
    def __init__(self, dim_state, dim_meas, V):
        self._dim_state = dim_state
        self._dim_meas = dim_meas
        self.V = V

    def f(self, x_est, dt):
        x_next = np.copy(x_est)
        return x_next

    def f_jacobian(self):
        G = np.eye(self._dim_state)
        return G  # dim [dim_state X dim_state]


class ProcessModelVelocities(ProcessModel):
    """ Simple point mass model,
        for derivation see scripts/matlab/point_mass_model"""
    def __init__(self, dim_state, dim_meas, V):
        super(ProcessModelVelocities, self).__init__(dim_state, dim_meas, V)

    def f(self, x_est, dt):
        self.x_next = np.copy(x_est)
        x = self.x_next[0]
        y = self.x_next[1]
        z = self.x_next[2]
        R = self.x_next[3]
        P = self.x_next[4]
        Y = self.x_next[5]
        dx = self.x_next[6]
        dy = self.x_next[7]
        dz = self.x_next[8]
        rr = self.x_next[9]
        pr = self.x_next[10]
        yr = self.x_next[11]

        # see matlab script
        c_1 = np.sin(R) * np.sin(Y) + np.cos(R) * np.cos(Y) * np.sin(P)
        c_2 = np.cos(R) * np.sin(Y) - np.cos(Y) * np.sin(P) * np.sin(R)
        c_3 = np.cos(P) * np.cos(Y)
        c_4 = np.cos(R) * np.cos(Y) + np.sin(P) * np.sin(R) * np.sin(Y)
        c_5 = np.cos(Y) * np.sin(R) - np.cos(R) * np.sin(P) * np.sin(Y)
        c_6 = np.cos(P) * np.sin(Y)
        c_7 = np.cos(P) * np.cos(R)
        c_8 = np.sin(P)
        c_9 = np.cos(P) * np.sin(R)
        delta_x = dt * (dz * c_1 - dy * c_2 + dx * c_3)
        delta_y = dt * (dy * c_4 - dz * c_5 + dx * c_6)
        delta_z = dt * (dz * c_7 - dx * c_8 + dy * c_9)
        delta_R = dt * (yr * c_1 - pr * c_2 + rr * c_3)
        delta_P = dt * (pr * c_4 - yr * c_5 + rr * c_6)
        delta_Y = dt * (yr * c_7 - rr * c_8 + pr * c_9)

        self.x_next[:3] = np.array([x + delta_x, y + delta_y, z + delta_z])
        self.x_next[3:6] = np.array([R + delta_R, P + delta_P, Y + delta_Y])

        self.G = np.array([
            [
                1, 0, 0,
                dt * (dy * c_1 + dz * c_2),
                delta_z * np.cos(Y),
                -delta_y,
                dt * c_3,
                -dt * c_2,
                dt * c_1,
                0, 0, 0
            ],
            [
                0, 1, 0,
                -dt * (dy * c_5 + dz * c_4),
                delta_z * np.sin(Y),
                delta_x,
                dt * c_6,
                dt * c_4,
                -dt * c_5,
                0, 0, 0
            ],
            [
                0, 0, 1,
                dt * (dy * c_7 - dz * c_9),
                -dt * (dx * np.cos(P) + np.sin(P) * (dz * np.cos(R) + dy * np.sin(R))),
                0,
                -dt * c_8,
                dt * c_9,
                dt * c_7,
                0, 0, 0
            ],
            [
                0, 0, 0,
                dt * (pr * c_1 + yr * c_2) + 1,
                delta_Y * np.cos(Y), -delta_P,
                0, 0, 0,
                dt * c_3,
                -dt * c_2,
                dt * c_1
            ],
            [
                0, 0, 0,
                -dt * (pr * c_5 + yr * c_4),
                dt * ((yr * np.cos(R) + pr * np.sin(R)) * c_6 - rr * c_8 * np.sin(Y)) + 1,
                delta_R,
                0, 0, 0,
                dt * c_6,
                dt * c_4,
                -dt * c_5
            ],
            [
                0, 0, 0,
                dt * (pr * c_7 - yr * c_9),
                -dt * (rr * np.cos(P) + c_8 * (yr * np.cos(R) + pr * np.sin(R))),
                1,
                0, 0, 0,
                -dt * c_8,
                dt * c_9,
                dt * c_7
            ],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
                     dtype=np.float)

        return self.x_next
    
    def f_jacobian(self):
        return self.G
