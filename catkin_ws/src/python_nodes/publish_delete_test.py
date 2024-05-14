"""Test script."""
# pylint: skip-file

import rospy
from visualization_msgs.msg import Marker, MarkerArray
import time

def main():
    # Initialize the ROS node
    rospy.init_node('marker_publisher')
    time.sleep(1)

    # Create a publisher
    pub = rospy.Publisher('/marker_test', MarkerArray, queue_size=10)
    time.sleep(1)

    # Create MarkerArray
    marker_array = MarkerArray()

    # Create a Marker and set its properties
    marker = Marker()
    marker.header.frame_id = "world"
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.pose.position.x = 1.0
    marker.pose.position.y = 1.0
    marker.pose.position.z = 1.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 5.1
    marker.scale.y = 5.1
    marker.scale.z = 5.1
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    # Add the Marker to the MarkerArray
    marker_array.markers.append(marker)

    # Publish the MarkerArray
    pub.publish(marker_array)
    rospy.loginfo("Published MarkerArray")

    # Wait for 5 seconds
    time.sleep(5)

    # Publish a DELETEALL action to remove all markers
    delete_marker = Marker()
    delete_marker.action = Marker.DELETEALL
    marker_array.markers = [delete_marker]
    pub.publish(marker_array)
    rospy.loginfo("Published DELETEALL")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
