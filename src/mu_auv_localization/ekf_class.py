from __future__ import print_function
import numpy as np
import rospy
import threading


class ExtendedKalmanFilter(object):
    def __init__(self, dim_state, dim_meas, measurement_model, process_model,
                 x0, p0_mat):
        self.dim_state = dim_state
        self.dim_meas = dim_meas
        self._x_est_0 = x0
        self._x_est = self._x_est_0
        self._x_est_last = self._x_est
        self._x_est_last_vision_step = None
        self._last_time_stamp_update = rospy.get_time()
        self._last_time_stamp_prediction = rospy.get_time()
        self._last_time_stamp_update_vision = rospy.get_time()
        self._last_time_stamp_update_pressure = rospy.get_time()
        self.last_pressure_meas = None
        self._p0_mat = p0_mat
        self._p_mat = self._p0_mat
        self.lock = threading.Lock()

        self.process_model = process_model
        self.measurement_model = measurement_model

    def get_x_est(self):
        return np.copy(self._x_est)

    def get_x_est_0(self):
        return np.copy(self._x_est_0)

    def get_p_mat(self):
        return np.copy(self._p_mat)

    def get_x_est_last(self):
        return np.copy(self._x_est_last)

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
        self._x_est_last = self._x_est
        self._x_est = self.process_model.f(self.get_x_est(), dt)
        a_mat = self.process_model.f_jacobian()
        self._p_mat = np.matmul(np.matmul(a_mat, self.get_p_mat()),
                                a_mat.transpose()) + self.process_model.V

        # reset EKF if unrealistic values
        if self.get_x_est()[0] > 5 or self.get_x_est()[1] > 5:
            print('Resetting EKF: x or y value outside tank')
            self.reset()
        # TODO find better values
        elif not np.all(np.isfinite(self.get_x_est())):
            print('Resetting EKF: unrealistically high value')
            print('x: ', self.get_x_est())
            self.reset()

        return True

    def update_vision_data(self, measurements, detected_tags, update_vel=True):
        # measurement is: dist, yaw to each tag

        self._x_est_last = self._x_est
        z_est_vision = self.measurement_model.h_vision_data(
            self.get_x_est(), detected_tags)
        h_mat_vision = self.measurement_model.h_jacobian_vision_data()
        w_mat_vision = self.measurement_model.vision_dynamic_meas_model(
            self.get_x_est(), measurements, detected_tags)

        y_vision = measurements - z_est_vision
        # wrap yaw innovation (every second entry) so it is between -pi and pi
        y_vision[1::2] = np.arctan2(np.sin(y_vision[1::2]),
                                    np.cos(y_vision[1::2]))

        # update pose
        self._x_est, self._p_mat = self._update(self.get_x_est(),
                                                self.get_p_mat(), y_vision,
                                                h_mat_vision, w_mat_vision)

        # update lin_vel
        if update_vel:
            if self._x_est_last_vision_step is None:
                self._last_time_stamp_update_vision = rospy.get_time()
                self._x_est_last_vision_step = self._x_est.copy()
                return True

            delta_t = max(0.1, rospy.get_time() - self._last_time_stamp_update_vision)
            vel = 1.0/(delta_t)*np.asarray(self._x_est[0:3, 0] - self._x_est_last_vision_step[0:3, 0]).reshape((-1, 1))

            y_vel = np.zeros((3, 1))
            for i in range(3):
                y_vel[i, 0] = min(0.7, max(-0.7, vel[i, 0] - self._x_est[i+6, 0]))

            h_mat_vel = self.measurement_model.h_jacobian_vel_data()
            w_mat_vel = self.measurement_model.w_mat_vel_data(detected_tags, delta_t)

            self._x_est, self._p_mat = self._update(self.get_x_est(),
                                                    self.get_p_mat(), y_vel,
                                                    h_mat_vel, w_mat_vel)
            self._last_time_stamp_update_vision = rospy.get_time()
            self._x_est_last_vision_step = self._x_est.copy()

        return True

    def update_orientation_data(self, measurements):
        # measurement is: roll, pitch from /mavros/local_position/pose

        z_est_orient = self.measurement_model.h_orientation_data(
            self.get_x_est())
        h_mat_orient = self.measurement_model.h_jacobian_orientation_data()

        y_orient = measurements - z_est_orient
        # wrap angle innovations between -pi and pi
        y_orient = np.arctan2(np.sin(y_orient), np.cos(y_orient))

        self._x_est, self._p_mat = self._update(
            self.get_x_est(), self.get_p_mat(), y_orient, h_mat_orient,
            self.measurement_model.w_mat_orientation)

        return True

    def update_imu_data(self, measurements, w_mat_imu):
        # measurement is either: body rates + lin. acceleration
        #               or: only body rates

        # check which measurements we're using:
        if measurements.shape[0] == 3:
            # only using angular velocities
            using_lin_acc = False
        elif measurements.shape[0] == 6:
            using_lin_acc = True
        else:
            print("IMU measurement has unexpected size!")
            using_lin_acc = False

        z_est_imu = self.measurement_model.h_imu_data(self.get_x_est(),
                                                      using_lin_acc)
        h_mat_imu = self.measurement_model.h_jacobian_imu_data()
        y_imu = measurements - z_est_imu
        self._x_est, self._p_mat = self._update(self.get_x_est(),
                                                self.get_p_mat(), y_imu,
                                                h_mat_imu, w_mat_imu)

        return True

    def update_pressure_data(self, measurement):
        """ innovation """
        if self.last_pressure_meas:
            z_est_pressure = self.measurement_model.h_pressure_data(self.get_x_est())
            h_mat_pressure = self.measurement_model.h_jacobian_pressure_data()
    
            delta_t = max(0.01, rospy.get_time() - self._last_time_stamp_update_pressure)
            self._last_time_stamp_update_pressure = rospy.get_time()
            dp_dt = (measurement - self.last_pressure_meas) / delta_t

            y_pressure = dp_dt - z_est_pressure

            if np.linalg.norm(y_pressure) > self.measurement_model.sat_press:
                y_pressure = (self.measurement_model.sat_press/np.linalg.norm(y_pressure)) * y_pressure

            w_mat_pressure = self.measurement_model.w_mat_pressure_data(delta_t)

            self._x_est, self._p_mat = self._update(self.get_x_est(),
                                                    self.get_p_mat(), y_pressure,
                                                    h_mat_pressure, w_mat_pressure)
            
        self.last_pressure_meas = measurement

        return True

    def _update(self, x_est, p_mat, y, h_mat, w_mat):
        """ helper function for general update """

        # compute K gain
        tmp = np.matmul(np.matmul(h_mat, p_mat), h_mat.transpose()) + w_mat
        k_mat = np.matmul(np.matmul(p_mat, h_mat.transpose()),
                          np.linalg.inv(tmp))

        # update state
        x_est = x_est + np.matmul(k_mat, y)

        # update covariance
        p_tmp = np.eye(self.dim_state) - np.matmul(k_mat, h_mat)
        p_mat = np.matmul(p_tmp, p_mat)
        # print('P_m diag: ', np.diag(self.get_p_mat()))
        return x_est, p_mat
