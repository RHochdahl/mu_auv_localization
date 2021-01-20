import numpy as np
from pyquaternion import Quaternion
import rospy


class ExtendedKalmanFilter(object):
    def __init__(self, x0=[1.0, 1.0, -0.5, 0, 0, 0]):  # states: x y z dx dy dz (world frame)
        """ initialize EKF """
        self.__x_est_0 = np.array([[x0[0]], [x0[1]], [x0[2]], [x0[3]], [x0[4]], [x0[5]]]).reshape((6, 1))
        self.__x_est = self.__x_est_0
        self.__x_est_last_step = self.__x_est

        self.__camera_pos = np.array([0.3, 0.0, -0.05])
        # rospy.loginfo("\n __init__:" + str(self.__x_est))
        self.__last_time_stamp_update_tags = rospy.get_time()
        self.__last_time_stamp_update_pressure = rospy.get_time()
        self.__last_time_stamp_prediction = rospy.get_time()
        # standard deviations
        self.__sig_x1 = 0.200
        self.__sig_x2 = 0.200
        self.__sig_x3 = 0.200
        self.__sig_x4 = 0.100
        self.__sig_x5 = 0.100
        self.__sig_x6 = 0.100
        self.__p_mat_0 = np.array(np.diag([self.__sig_x1 ** 2,
                                           self.__sig_x2 ** 2,
                                           self.__sig_x3 ** 2,
                                           self.__sig_x4 ** 2,
                                           self.__sig_x5 ** 2,
                                           self.__sig_x6 ** 2]))
        self.__p_mat = self.__p_mat_0
        # process noise
        self.__sig_w1 = 0.05
        self.__sig_w2 = 0.05
        self.__sig_w3 = 0.03
        self.__sig_w4 = 0.05
        self.__sig_w5 = 0.05
        self.__sig_w6 = 0.05
        self.__q_mat = np.array(np.diag([self.__sig_w1 ** 2,
                                         self.__sig_w2 ** 2,
                                         self.__sig_w3 ** 2,
                                         self.__sig_w4 ** 2,
                                         self.__sig_w5 ** 2,
                                         self.__sig_w6 ** 2]))

        # measurement noise
        # --> see measurement_covariance_model
        self.__r_tags_0 = 0.1
        self.__r_tags_lin_fac = 0.01
        # measurement noise velocity
        self.__sig_v = 0.5
        self.__v_mat = np.array(np.diag([self.__sig_v ** 2, self.__sig_v ** 2, self.__sig_v ** 2]))

        self.yaw_current = 0
        self.pitch_current = 0
        self.roll_current = 0
        self.__counter_not_seen_any_tags = 0
        # initial values and system dynamic (=eye)
        self.__f_mat = np.eye(6)  # not used

        self.pascal_per_meter = 9.78057e3  # g*rho
        self.last_pressure_meas = None
        self.__sig_dz = 5.0

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
        self.__x_est = x0
        # rospy.loginfo("\n set_x_0:" + str(self.__x_est))
        return True

    def set_p_mat_0(self, p0):
        self.__p_mat = p0
        return True

    def reset_ekf(self):
        self.__x_est = self.__x_est_0
        # rospy.loginfo("\n reset_ekf:" + str(self.__x_est))
        self.__p_mat = self.__p_mat_0

    def get_x_est(self):
        tmp = np.copy(self.__x_est)
        # rospy.loginfo("\n get_x_est:" + str(self.__x_est))
        # tmp[0] = tmp[0]  # (self.__x_est[2]*0.04+1)*tmp[0]-0.05
        # tmp[1] = tmp[1]
        # tmp[2] = tmp[2]
        return tmp.round(decimals=3)

    def get_p_mat(self):
        return self.__p_mat

    def get_z_meas(self):
        return self.__z_meas

    def get_y_est(self):
        return self.__y_est

    def get_roll_current(self):
        return self.roll_current

    def get_pitch_current(self):
        return self.pitch_current

    def get_yaw_current(self):
        return self.yaw_current

    # measurement function
    def h_tags(self, x, vis_tags):
        num_vis_tags = vis_tags.shape[0]
        z = np.zeros((num_vis_tags, 1))
        h_jac = np.zeros((num_vis_tags, 3))

        rotation = self.yaw_pitch_roll_to_quat(self.yaw_current, self.pitch_current, self.roll_current)
        camera_offset = rotation.rotate(self.__camera_pos)

        for i, tag in enumerate(vis_tags):
            tag_pos = tag[1:4]
            # print("tag pos + " + str(tag_pos))
            # print("x= " + str(x))
            # r = sqrt((x - x_tag) ^ 2 + (y - y_tag) ^ 2 + (z - z_tag) ^ 2)
            r_dist = np.sqrt((x[0] + camera_offset[0] - tag_pos[0]) ** 2 +
                             (x[1] + camera_offset[1] - tag_pos[1]) ** 2 +
                             (x[2] + camera_offset[2] - tag_pos[2]) ** 2)
            # print ("r = " + str(r_dist))
            z[i, 0] = r_dist

            # dh /dx1
            h_jac_x1 = (1.0/r_dist) * (x[0] - tag_pos[0])
            # dh /dx2
            h_jac_x2 = (1.0/r_dist) * (x[1] - tag_pos[1])
            # dh /dx3
            h_jac_x3 = (1.0/r_dist) * (x[2] - tag_pos[2])

            h_jac[i, 0:3] = [h_jac_x1, h_jac_x2, h_jac_x3]

        num_vis_tags = vis_tags.shape[0]

        return z, h_jac

    def current_rotation(self, yaw_current, pitch_current, roll_current):
        self.yaw_current = yaw_current
        self.pitch_current = pitch_current
        self.roll_current = roll_current

    def update_velocity_if_nothing_is_seen(self):
        self.__x_est[3:6] = 0.9*self.__x_est[3:6]
        self.__counter_not_seen_any_tags = self.__counter_not_seen_any_tags + 1

    def prediction(self, x_rot_vel, y_rot_vel, z_rot_vel):
        """ prediction """
        # accel_x=0

        delta_t = rospy.get_time() - self.__last_time_stamp_prediction
        self.__last_time_stamp_prediction = rospy.get_time()
        if delta_t == 0:
            delta_t = 0.02
        # print("current_time:", rospy.get_time())
        # print("delta_t", delta_t)
        self.yaw_current = self.yaw_current - z_rot_vel * delta_t
        self.pitch_current = self.pitch_current - y_rot_vel * delta_t  # nicht sicher ob das richtig ist
        self.roll_current = self.roll_current + x_rot_vel * delta_t  # nicht sicher ob das richtig ist
        # rotation = self.yaw_pitch_roll_to_quat(self.yaw_current, self.pitch_current, self.roll_current)
        # update_x_y_z = np.asarray([[self.__x_est[3] * delta_t], [self.__x_est[4] * delta_t], [self.__x_est[5] * delta_t]])  # yaw_pitch_roll_to_quat
        # update_x_y_z = rotation.rotate(np.asarray([[0], [0], [0]]))
        # update_x_y_z_v = np.asarray([[update_x_y_z[0]], [update_x_y_z[1]], [update_x_y_z[2]], [0], [0], [0]])
        # print(update_x_y_z_v)

        # self.__x_est[0:3] = self.__x_est[0:3] + delta_t * self.__x_est[1:4]  # prediction = f * x_est + u
        
        # rospy.loginfo("\n prediction:" + str(self.__x_est))
        self.__p_mat = np.matmul(self.__f_mat, np.matmul(self.__p_mat, np.transpose(self.__f_mat))) + self.__q_mat
        # print(self.__p_mat)
        return True

    def update_tags(self, z_meas_tags):
        """ innovation """
        num_meas = z_meas_tags.shape[0]
        # get new measurement
        z_meas = z_meas_tags[:, 0].reshape(num_meas, 1)
        # print("z_meas", np.transpose(z_meas.round(decimals=3)))
        # estimate measurement from x_est
        z_est, h_jac_mat = self.h_tags(self.__x_est[0:3], z_meas_tags)
        z_tild = z_meas - z_est
        # print("z_est", np.transpose(z_est.round(decimals=3)))
        # print("z_tild", np.transpose(z_tild.round(decimals=3)))
        # calc K-gain
        # print("h_jac_mat", h_jac_mat)
        # k_mat = np.zeros((3, num_meas))
        # r_mat_temp = np.eye(num_meas) * self.__r_mat  # same measurement noise for all measurements, for the moment
        r_mat = np.zeros([num_meas, num_meas])
        for i, h in enumerate(z_est):
            r_mat[i, i] = self.__r_tags_0+self.__r_tags_lin_fac*h
        s_mat = np.linalg.inv(np.linalg.multi_dot([h_jac_mat, self.__p_mat[0:3, 0:3], h_jac_mat.transpose()]) + r_mat)
        k_mat = np.linalg.multi_dot([self.__p_mat[0:3, 0:3], h_jac_mat.transpose(), s_mat])
        # s_mat = np.dot(h_jac_mat, np.dot(self.__p_mat[0:3, 0:3], h_jac_mat.transpose())) + r_mat_temp
        # print("s_mat", s_mat)
        # s_diag = np.diag(s_mat)
        # compute k_mat in an iterative way
        # for i_tag in range(num_meas):
        #    k_mat[:, i_tag] = np.dot(self.__p_mat[0:3, 0:3], h_jac_mat[i_tag, :].transpose()) / s_diag[i_tag]  # 1/s scalar since s_mat is dim = 1x1
        # check distance to tag and reject far away tags
        # print("k_mat", k_mat)
        # print("bebfore update x_est:", self.__x_est)
        self.__x_est[0:3] = self.__x_est[0:3] + np.matmul(k_mat, z_tild).reshape((3, 1))  # = x_est + k * z_tild
        # rospy.loginfo("\n update_tags:" + str(self.__x_est))
        # print("after update x_est:", self.__x_est)
        # velocity calculation:
        # innovation y=z-h(x)

        # angle_velocity = np.arctan2(self.__x_est[1] - self.__x_est_last_step[1],
        #                            self.__x_est[0] - self.__x_est_last_step[0])

        # scaling = np.cos(
        #    np.arctan2(np.sin(angle_velocity - self.yaw_current), np.cos(angle_velocity - self.yaw_current)))
        # if abs(np.arctan2(np.sin(angle_velocity - self.yaw_current),
        #                  np.cos(angle_velocity - self.yaw_current))) > np.pi / 2:
        #    scaling = 0
        # print(scaling)
        delta_t = rospy.get_time() - self.__last_time_stamp_update_tags
        if delta_t == 0:
            delta_t = 0.1
        vel = 1.0/(delta_t*(self.__counter_not_seen_any_tags+1))*np.asarray(self.__x_est[0:3] - self.__x_est_last_step[0:3])   # * scaling
        if self.__counter_not_seen_any_tags > 0:
            self.__counter_not_seen_any_tags = self.__counter_not_seen_any_tags - 1

        p_update = np.zeros((6, 6))
        for i in range(3):
            vel[i] = min(0.7, max(-0.7, vel[i]))
            y_vel = vel - self.__x_est[i+3]
            # gain_vel_covarianz=(1+abs(np.linalg.norm(self.__x_est[0:3] - self.__x_est_last_step[0:3]))*100)
            # print(gain_vel_covarianz)
            s_k_mat = self.__p_mat[i+3, i+3] + self.__v_mat[i]
            kalman_gain_v = self.__p_mat[i+3, i+3] / s_k_mat
            # self.__p_mat[3,3] = np.matmul((np.eye(3)-kalman_gain_v),self.__p_mat[3,3])
            self.__x_est[i+3] = min(0.7, max(-0.7, self.__x_est[i+3] + kalman_gain_v* y_vel))
            # rospy.loginfo("\n update_tags:" + str(self.__x_est))
            p_update[i+3, i+3] = (1 - kalman_gain_v)
        p_update[0:3, 0:3] = (np.eye(3) - np.matmul(k_mat, h_jac_mat))
        self.__p_mat = np.matmul(p_update, self.__p_mat)
        # if z_vel > 1:
        # print("vel to high:", z_vel,delta_t)
        #    self.__x_est[3] = 0
        # print("after estimation")
        # print(self.__x_est)
        # save last state
        self.__x_est_last_step = np.copy(self.__x_est)
        self.__last_time_stamp_update_tags = rospy.get_time()

        if self.__x_est[0] > 5 or np.isnan(self.__x_est[0]) or self.__x_est[0] < -1:
            self.__x_est[0] = 1.5
            self.__x_est_last_step[0] = self.__x_est[0]
            self.__p_mat[0] = self.__p_mat_0[0]
        if self.__x_est[1] > 3 or np.isnan(self.__x_est[1]) or self.__x_est[1] < -1:
            self.__x_est[1] = 1
            self.__x_est_last_step[1] = self.__x_est[1]
            self.__p_mat[1] = self.__p_mat_0[1]

        if self.__x_est[2] > 1.5 or np.isnan(self.__x_est[2]) or self.__x_est[2] < -0.2:
            self.__x_est[2] = 0.5
            self.__x_est_last_step[2] = self.__x_est[2]
            self.__p_mat[2] = self.__p_mat_0[2]
        if self.__x_est[3] > 1:
            self.__x_est[3] = 1
        # rospy.loginfo("\n update_tags:" + str(self.__x_est))

        return True

    def update_pressure(self, z_meas_pressure):
        """ innovation """
        if self.last_pressure_meas:
            delta_t = max(0.01, rospy.get_time() - self.__last_time_stamp_update_pressure)
            z_vel = -(z_meas_pressure - self.last_pressure_meas) / (delta_t*self.pascal_per_meter)
            z_vel = min(0.7, max(-0.7, z_vel))
            y_vel = z_vel - self.__x_est[5]
            s_k_mat = self.__p_mat[5, 5] + self.__sig_dz
            kalman_gain_v = self.__p_mat[5, 5] / s_k_mat
            self.__x_est[5] = self.__x_est[5] + kalman_gain_v * y_vel
            self.__x_est[5] = min(0.7, max(-0.7, self.__x_est[5]))
            p_update = (1 - kalman_gain_v)
            self.__p_mat[5, 5] = p_update * self.__p_mat[5, 5]
            self.__last_time_stamp_update_pressure = rospy.get_time()
            # rospy.loginfo("\n update_pressure:" + str(self.__x_est))
        self.last_pressure_meas = z_meas_pressure

        return True
