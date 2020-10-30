#!/usr/bin/env python

import yaml
import os
import tf.transformations


def generate_even_grid():
    tag_count = 63
    data = {}
    data["tag_bundles"] = []
    bundle = {}
    bundle["name"] = "gazebo_bundle"
    bundle["layout"] = []
    for i in range(9):
        for j in range(7):
            # use gazebo coordinate frame
            tag_id = i * 7 + j
            pos_x = (6 - j) * 0.25 + 0.06
            pos_y = i * 0.39375 + 0.06
            pos_z = -1.3
            # order of quaternion is x, y, z, w
            quat = tf.transformations.quaternion_from_euler(0, 0, 0)
            bundle["layout"].append({
                "id": tag_id,
                "size": 0.096,
                "x": pos_x,
                "y": pos_y,
                "z": pos_z,
                "qx": float(quat[0]),
                "qy": float(quat[1]),
                "qz": float(quat[2]),
                "qw": float(quat[3]),
            })
    data["tag_bundles"].append(bundle)
    return data


def main():
    filename = "tags.yaml"
    with open(filename, "w") as file_handle:
        data = generate_even_grid()
        yaml.dump(data, file_handle)
        print("Created file '{}'".format(os.path.join(os.getcwd(), filename)))
        print("Probably you want to move it to this package's config directory")


if __name__ == "__main__":
    main()
