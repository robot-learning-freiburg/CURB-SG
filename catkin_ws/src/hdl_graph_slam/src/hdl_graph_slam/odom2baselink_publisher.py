#!/usr/bin/python
# SPDX-License-Identifier: BSD-2-Clause
# pylint: disable=wildcard-import, attribute-defined-outside-init, unused-wildcard-import,
# pylint: disable=missing-module-docstring, missing-class-docstring, duplicate-code
import time

import rospy
import tf
from geometry_msgs.msg import *


class Map2OdomPublisher:
    def __init__(self):
        self.broadcaster = tf.TransformBroadcaster()
        self.agent_no = rospy.get_param("/agent_no")

        print(f"Odom2BaseLink Publisher Agent No: {self.agent_no}")
        self.subscriber = rospy.Subscriber(
            f"/scan_matching_odometry_{self.agent_no}/transform",
            TransformStamped,
            self.callback,
        )

    def callback(self, odom_msg):

        pose = odom_msg.transform
        pos = (pose.translation.x, pose.translation.y, pose.translation.z)
        quat = (pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w)

        map_frame_id = odom_msg.header.frame_id
        odom_frame_id = odom_msg.child_frame_id

        self.broadcaster.sendTransform(
            pos,
            quat,
            # self.odom_msg.header.stamp,
            odom_msg.header.stamp,
            odom_frame_id,
            map_frame_id,
        )

    def spin(self):
        pass


def main():
    rospy.init_node("odom2base_link_publisher")
    node = Map2OdomPublisher()

    # rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        # rate.sleep()
        time.sleep(1.0)


if __name__ == "__main__":
    main()
