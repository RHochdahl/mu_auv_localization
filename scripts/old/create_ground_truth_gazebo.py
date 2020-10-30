import numpy as np


def callback_gantry():
    """"""
    # id,x,y,z,N,seen
    number_of_tags = 63
    array_tags = np.zeros((number_of_tags, 5))
    for i in range(9):
        for j in range(7):
            array_tags[i * 7 + j, 0] = i * 7 + j
            array_tags[i * 7 + j, 1] = i*0.39375 + 0.06
            array_tags[i * 7 + j, 2] = (6-j)*0.25 + 0.06
            array_tags[i * 7 + j, 3] = 1.3

    print(array_tags)
    return array_tags
    offset_to_top_of_water = 0


def main():
    global mean_array, number_of_tags
    mean_array = callback_gantry()
    np.savetxt("calibration_ground_truth_gazebo.csv", mean_array, delimiter=',')
    print(mean_array)


if __name__ == '__main__':
    main()
