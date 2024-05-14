"""Helper script to write the received bounding boxes into a file."""
# pylint: skip-file

import csv

import rospy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray


def callback(data):
    center_points = []
    for box in data.boxes:
        if box.label == 1:
            center = box.pose.position
            center_points.append((center.x, center.y))

    with open('signs_center_points.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['X', 'Y'])
        for point in center_points:
            writer.writerow([point[0], point[1]])

    print("File written. ")
    exit(0)


def main():
    rospy.init_node('bounding_box_center')
    rospy.Subscriber("/feature_bounding_boxes", BoundingBoxArray, callback)
    print("Init done. ")
    rospy.spin()


if __name__ == '__main__':
    main()
