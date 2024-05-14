#!/usr/bin/python
# SPDX-License-Identifier: BSD-2-Clause
# pylint: disable=wildcard-import, attribute-defined-outside-init, unused-wildcard-import,
# pylint: disable=missing-module-docstring, missing-class-docstring, duplicate-code
import rospy
import tf
from geometry_msgs.msg import *


class Odom2BaselinkPublisher:
    def __init__(self):
        self.broadcaster = tf.TransformBroadcaster()
        self.agent_no = rospy.get_param("/agent_no")

        print(f"Map2OdomPublisher Agent No: {self.agent_no}")
        self.subscriber = rospy.Subscriber(
            f"/hdl_graph_slam_{self.agent_no}/odom2map", TransformStamped, self.callback
        )

    def callback(self, odom_msg):
        self.odom_msg = odom_msg

    def spin(self):
        if not hasattr(self, "odom_msg"):
            self.broadcaster.sendTransform(
                (0, 0, 0),
                (0, 0, 0, 1),
                rospy.Time.now(),
                f"odom_{self.agent_no}",
                f"map_{self.agent_no}",
            )
            return

        pose = self.odom_msg.transform
        pos = (pose.translation.x, pose.translation.y, pose.translation.z)
        quat = (pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w)

        map_frame_id = self.odom_msg.header.frame_id
        odom_frame_id = self.odom_msg.child_frame_id

        self.broadcaster.sendTransform(
            pos,
            quat,
            # self.odom_msg.header.stamp,
            self.odom_msg.header.stamp,
            odom_frame_id,
            map_frame_id,
        )


def main():
    rospy.init_node("map2odom_publisher")
    node = Odom2BaselinkPublisher()

    # rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        # rate.sleep()


if __name__ == "__main__":
    main()
