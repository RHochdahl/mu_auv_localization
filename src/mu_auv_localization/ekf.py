# DO NOT USE CURRENTLY. THIS IS STILL WORK IN PROGRESS.
import numpy as np
import tf.transformations
import rospy


class ExtendedKalmanFilter(object):
    def __init__(self, x0, P_0=None, Q=None, R=None):
        self._x_0 = np.array(x0, dtype=np.float).reshape((-1, 1))
        self._x = self._x_0
        self._x_last = self._x
        self._stamps = dict(last_update=0.0, last_predict=0.0)

        if P_0:
            self._P_0 = P_0
        else:
            self._P_0 = np.diag([100] * len(self._x_0))

        self._P = self._P_0

        if Q:
            self._Q = Q
        else:
            self._Q = np.diag([0.01] * len(self._x_0))

        if R:
            self._R = R
        else:
            self._R = np.diag([1, 1, 1, 1, 1, 1, 1])

    def h_fun(self, x):
        return x[0:7]

    def h_fun_jacobian(self):
        cols = len(self._x)
        ret = np.zeros((7, cols))
        ret[:, :7] = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        return ret

    def f_fun(self, x, dt, omega_gyro):
        position = x[0:3]
        orientation = x[3:7]
        velocity = x[7:]
        # compute new orientation

        # see http://stanford.edu/class/ee267/lectures/lecture10.pdf
        omega_gyro_norm = np.linalg.norm(omega_gyro)
        if omega_gyro_norm > 0.0001:
            omega_gyro_normalized = omega_gyro / omega_gyro_norm
            q_delta = tf.transformations.quaternion_about_axis(
                dt * omega_gyro_norm, omega_gyro_normalized)
            q_new = tf.transformations.quaternion_multiply(orientation, q_delta)
        else:
            q_new = orientation

        # compute new position
        pos_new = position + dt * velocity

        return np.concatenate(
            [pos_new.ravel(), q_new.ravel(),
             velocity.ravel()]).reshape(-1, 1)

    def f_fun_jacobian(self, dt):
        return np.array(
            [[1, 0, 0, 0, 0, 0, 0, dt, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, dt, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, dt], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
            dtype=np.float)

    def predict(self, dt, omega_gyro):
        self._x = self.f_fun(self._x, dt, omega_gyro)
        F = self.f_fun_jacobian(dt)
        self._P = np.matmul(F, self._P)
        self._P = np.matmul(self._P, F.transpose()) + self._Q

    def update(self, dt, z):
        y = z - self.h_fun(self._x)
        H = self.h_fun_jacobian()
        S_helper = np.matmul(H, self._P)
        S = np.matmul(S_helper, H.transpose()) + self._R
        K = np.matmul(self._P, H.transpose())
        K = np.matmul(K, np.linalg.inv(S))
        self._x = self._x + np.matmul(K, y)
        P = np.eye(len(K)) - np.matmul(K, H)
        self._P = np.matmul(P, self._P)

    def get_covariance_6d(self):
        # see www.ucalgary.ca/engo_webdocs/GL/96.20096.JSchleppe.pdf for implementation
        pass


def velocity(t):
    vx0 = 0.0
    vy0 = 0.0
    vz0 = 0.0
    vx = vx0 + np.sin(10 * t) + 0.01 * t
    vy = vy0 + 0.01 * t
    vz = vz0 + t**2
    return np.stack([vx, vy, vz], axis=-1)


def position(velocity, t):
    sx0 = 0
    sy0 = 0
    sz0 = 0
    sx = np.zeros([len(t), 1])
    sy = np.zeros_like(sx)
    sz = np.zeros_like(sx)
    dt = np.diff(t)
    sx[0, 0] = sx0
    sy[0, 0] = sy0
    sz[0, 0] = sz0
    sx[1:, 0] = dt * velocity[:, 0] + sx


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    vx = 0.1
    vy = 0.1
    vz = 0.1

    x0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, vx, vy, vz],
                  dtype=np.float).reshape(-1, 1)
    steps = 1000
    dt = 0.01
    t = [0]
    truth = np.zeros((len(x0), steps))
    measurements = np.zeros((7, steps))
    truth[:, 0:1] = x0
    measurements[:, 0:1] = x0[:7]

    ekf = ExtendedKalmanFilter(np.zeros_like(x0))
    estimations = np.zeros_like(truth)
    P = np.zeros_like(estimations)

    for i in range(1, steps):
        t.append(t[i - 1] + 0.01)
        truth[:,
              i:i + 1] = truth[:, i - 1:i] + np.array(
                  [dt * vx, dt * vy, dt * vz, 0, 0, 0, 0, 0, 0, 0],
                  dtype=np.float).reshape(-1, 1)
        x_var = (np.random.rand() - 0.5) * 1
        y_var = (np.random.rand() - 0.5) * 1
        z_var = (np.random.rand() - 0.5) * 1
        qx_var = (np.random.rand() - 0.5) * 0.02
        qy_var = (np.random.rand() - 0.5) * 0.02
        qz_var = (np.random.rand() - 0.5) * 0.02
        qw_var = (np.random.rand() - 0.5) * 0.02
        quat = truth[3:7, i]
        quat += np.array([qx_var, qy_var, qz_var, qw_var], dtype=np.float)
        quat = quat / np.linalg.norm(quat)
        pos = np.array([x_var, y_var, z_var], dtype=np.float)
        noise = np.concatenate([pos, quat]).reshape(-1, 1)

        measurement = truth[:7, i:i + 1] + noise
        ekf.predict(dt, np.array([0, 0, 0], dtype=np.float))
        if i % 10 == 0:
            ekf.update(dt, measurement)
        estimations[:, i:i + 1] = ekf._x
        P[:, i:i + 1] = np.diagonal(ekf._P).reshape(-1, 1)
        measurements[:, i:i + 1] = measurement

    plt.plot(t, measurements[0, :], t, truth[0, :], t, estimations[0, :], "r")
    plt.plot(t, P[0, :])
    plt.show()
