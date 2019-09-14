import numpy as np


class ExtendedKalmanFilter(object):
    def __init__(self, x0=[1.0, 1.0, 0.5]):

        """ initialize EKF """
        self.__x_est_0 = np.array([[x0[0]], [x0[1]], [x0[2]]]).reshape((3, 1))
        self.__x_est = self.__x_est_0
        # standard deviations
        self.__sig_x1 = 1.500
        self.__sig_x2 = 1.500
        self.__sig_x3 = 1.500

        self.__p_mat_0 = np.array(np.diag([self.__sig_x1 ** 2,
                                           self.__sig_x2 ** 2,
                                           self.__sig_x3 ** 2]))
        self.__p_mat = self.__p_mat_0
        # process noise
        self.__sig_w1 = 0.500
        self.__sig_w2 = 0.500
        self.__sig_w3 = 0.300
        self.__q_mat = np.array(np.diag([self.__sig_w1 ** 2,
                                         self.__sig_w2 ** 2,
                                         self.__sig_w3 ** 2]))

        # measurement noise
        # --> see measurement_covariance_model
        self.__sig_r = 1
        self.__r_mat = self.__sig_r ** 2

        # initial values and system dynamic (=eye)
        self.__i_mat = np.eye(3)

        # self.__z_meas = np.zeros(self.__max_tag)
        # self.__y_est = np.zeros(self.__max_tag)
        # self.__r_dist = np.zeros(self.__max_tag)

    def set_x_0(self, x0):
        self.__x_est = x0
        return True

    def set_p_mat_0(self, p0):
        self.__p_mat = p0
        return True

    def reset_ekf(self):
        self.__x_est = self.__x_est_0
        self.__p_mat = self.__p_mat_0

    def get_x_est(self):
        return self.__x_est.round(decimals=3)

    def get_p_mat(self):
        return self.__p_mat

    def get_z_meas(self):
        return self.__z_meas

    def get_y_est(self):
        return self.__y_est

    # measurement function
    def h(self, x, vis_tags):
        num_vis_tags = vis_tags.shape[0]
        z = np.zeros((num_vis_tags, 1))

        for i, tag in enumerate(vis_tags):
            tag_pos = tag[1:4]
            # print("tag pos + " + str(tag_pos))
            # print("x= " + str(x))
            # r = sqrt((x - x_tag) ^ 2 + (y - y_tag) ^ 2 + (z - z_tag) ^ 2)
            r_dist = np.sqrt((x[0] - tag_pos[0]) ** 2 +
                             (x[1] - tag_pos[1]) ** 2 +
                             (x[2] - tag_pos[2]) ** 2)
            #print ("r = " + str(r_dist))
            z[i, 0] = r_dist

        return z

    # Jacobian of the measurement function
    def h_jacobian(self, x, vis_tags):

        num_vis_tags = vis_tags.shape[0]
        h_jac = np.zeros((num_vis_tags, 3))

        for i, tag in enumerate(vis_tags):
            tag_pos = tag[1:4]
            # r = sqrt((x - x_tag) ^ 2 + (y - y_tag) ^ 2 + (z - z_tag) ^ 2)
            r_dist = np.sqrt((x[0] - tag_pos[0]) ** 2 +
                             (x[1] - tag_pos[1]) ** 2 +
                             (x[2] - tag_pos[2]) ** 2)

            # dh /dx1
            h_jac_x1 = 0.5 * 2.0 * (x[0] - tag_pos[0]) / r_dist
            # dh /dx2
            h_jac_x2 = 0.5 * 2.0 * (x[1] - tag_pos[1]) / r_dist
            # dh /dx3
            h_jac_x3 = 0.5 * 2.0 * (x[2] - tag_pos[2]) / r_dist

            h_jac[i, 0:3] = [h_jac_x1, h_jac_x2, h_jac_x3]

        return h_jac  # dim [num_tag X 3]

    def prediction(self):
        """ prediction """
        self.__x_est = self.__x_est  # + np.random.randn(3, 1) * 1  # = I * x_est
        self.__p_mat = self.__i_mat.dot(self.__p_mat.dot(self.__i_mat)) + self.__q_mat
        return True

    def update(self, z_meas_tags):
        """ innovation """
        num_meas = z_meas_tags.shape[0]
        # get new measurement
        z_meas = z_meas_tags[:, 0].reshape(num_meas, 1)

        # iterate through all tx-rss-values
        # for itx in range(self.__tx_num):
        # estimate measurement from x_est
        z_est = self.h(self.__x_est, z_meas_tags)
        z_tild = z_meas - z_est
        #print("zmeas = " + str(z_meas.transpose()))
        #print("yest = " + str(y_est.transpose()))
        #print("ytild = " + str(y_tild.transpose()))
        # calc K-gain
        h_jac_mat = self.h_jacobian(self.__x_est, z_meas_tags)

        # iterate through measurements for serial update

        for i_tag in range(num_meas):
            if (z_meas[i_tag] > 1.5):
                #print("skiptag:",z_meas[i_tag])
                continue

            h_jac_red = h_jac_mat[i_tag, 0:3].reshape((1,3))  # [tag X 3]--> [1x3]
            r_red = self.__r_mat  # already 1x1
            z_tild_red = z_tild[i_tag, 0]
            #print(i_tag)
            #print(y_tild_red)
            s_mat = np.dot(h_jac_red, np.dot(self.__p_mat, h_jac_red.transpose())) + r_red  # = H * P * H^t + R
            # print("smat = " + str(s_mat))

            k_mat = np.dot(self.__p_mat, h_jac_red.transpose()) / s_mat  # 1/s scalar since s_mat is dim = 1x1
            #print(i_tag)
            #print("k_mat = " + str(k_mat.transpose()))
            #print("y_tild = " + str(y_tild_red))
            self.__x_est = self.__x_est + k_mat * z_tild_red  # = x_est + k * y_tild
            # self.__x_est[2, 0] = np.abs(self.__x_est[2, 0])
            self.__p_mat = (self.__i_mat - np.dot(k_mat, h_jac_red)) * self.__p_mat  # = (I-KH)*P

        #print("x_up= " + str(self.__x_est.transpose()))
        return True

