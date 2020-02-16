from apriltag_ros.msg import AprilTagDetectionArray
from apriltag_ros.msg import AprilTagDetection
from gantry_control_ros.msg import gantry
import numpy as np

import rospy
import tf
from pyquaternion import Quaternion

# id,x,y,z,N,seen
number_of_tags = 63
array_tags = np.zeros((number_of_tags, 6))
for i in range(number_of_tags):
    array_tags[i, 0] = i
print(array_tags)
offset_to_top_of_water = 0


def callback_april(msg):
    """"""
    global array_tags, number_of_tags
    num_meas = len(msg.detections)
    what_was_in_msg = list()
    if num_meas >= 1:
        for i, tag in enumerate(msg.detections):
            tag_id = int(tag.id[0])
            what_was_in_msg.append(tag_id)
            array_tags[tag_id, 1] = tag.pose.pose.pose.position.x
            array_tags[tag_id, 2] = tag.pose.pose.pose.position.y
            array_tags[tag_id, 3] = tag.pose.pose.pose.position.z + offset_to_top_of_water

    for i in range(number_of_tags):
        if i in what_was_in_msg:
            array_tags[i, 5] = 1
        else:
            array_tags[i, 5] = 0
    # print(array_tags)
    pass


mean_array = np.zeros((number_of_tags, 5))
for i in range(number_of_tags):
    mean_array[i, 0] = i


def callback_gantry(msg):
    """"""
    global array_tags, mean_array, number_of_tags
    currently_seen = array_tags[np.where(array_tags[:, 5] == 1), :]
    qz_90n = Quaternion(axis=[0, 0, 1], angle=np.pi / 2)
    for i in range(currently_seen[0].shape[0]):
        #print(currently_seen[0, i, 1:4])
        currently_seen[0, i, 1:4] = qz_90n.rotate(currently_seen[0, i, 1:4])
        #print(currently_seen[0, i, 1:4])
        mean_array[int(currently_seen[0, i, 0]), 1] = mean_array[int(currently_seen[0, i, 0]), 1] + currently_seen[
            0, i, 1] + msg.pos_gantry.x
        mean_array[int(currently_seen[0, i, 0]), 2] = mean_array[int(currently_seen[0, i, 0]), 2] + currently_seen[
            0, i, 2] + msg.pos_gantry.y
        mean_array[int(currently_seen[0, i, 0]), 3] = mean_array[int(currently_seen[0, i, 0]), 3] + currently_seen[
            0, i, 3] + msg.pos_gantry.z
        mean_array[int(currently_seen[0, i, 0]), 4] = mean_array[int(currently_seen[0, i, 0]), 4] + 1
    # print(mean_array)
    pass


def main():
    global mean_array, number_of_tags
    rospy.init_node('particle_filter_node')
    rospy.Subscriber("/tag_detections", AprilTagDetectionArray, callback_april, queue_size=1)
    rospy.Subscriber("/gantry/current_position", gantry, callback_gantry, queue_size=1)

    while not rospy.is_shutdown():
        pass
    print(mean_array)
    for i in range(number_of_tags):
        mean_array[i, 1:5] = mean_array[i, 1:5] / float(mean_array[i, 4])
    np.savetxt("calibration.csv", mean_array, delimiter=',')
    print(mean_array)


if __name__ == '__main__':
    main()