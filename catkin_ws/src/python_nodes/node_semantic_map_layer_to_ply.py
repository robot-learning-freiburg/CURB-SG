"""This node subscribes to a pointcloud topic, and writes the content 
into a ply file to view it in a differnt program or so."""
# pylint: skip-file
import os

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


def callback(data):
    points_list = []

    # Iterate over the PointCloud2 data
    for point in pc2.read_points(data, skip_nans=True):
        x, y, z = point[:3]
        semantic_class = point[6]

        # Filter points with semantic class 7
        if semantic_class == 7:
            points_list.append([x, y, z])

    # Convert to NumPy array and save as .ply file
    points_array = np.array(points_list, dtype=np.float32)
    save_ply_file(points_array, 'filtered_points.ply')


def save_ply_file(points, filename):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for point in points:
            f.write("{} {} {}\n".format(point[0], point[1], point[2]))


def main():
    rospy.init_node('filter_points_node', anonymous=True)
    rospy.Subscriber("/map_server/map_points", PointCloud2, callback)
    rospy.spin()


if __name__ == '__main__':
    main()
