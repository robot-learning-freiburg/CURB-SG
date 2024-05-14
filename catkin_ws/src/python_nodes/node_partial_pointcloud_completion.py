"""
collecting ideas for this task here
"""

# pylint: disable=import-error, undefined-variable

import numpy as np
import rospy
from registration import cpd
from sensor_msgs.msg import PointCloud


def callback(partial_clouds):
    """Callback"""
    # Convert the ROS message to numpy arrays
    partial_clouds_np = []
    for partial_cloud in partial_clouds:
        partial_cloud_np = np.array(partial_cloud.points).reshape(-1, 3)
        partial_clouds_np.append(partial_cloud_np)

    # Align the partial point clouds to a reference point cloud using CPD
    reference_cloud = partial_clouds_np[0]
    aligned_clouds = []
    for partial_cloud_np in partial_clouds_np:
        _, aligned_cloud_np, _, _ = cpd.registration(partial_cloud_np,
                                                     reference_cloud,
                                                     method='rigid')

        # Add the aligned partial point cloud to the list of aligned point clouds
        aligned_clouds.append(aligned_cloud_np)

    # Merge the aligned partial point clouds into a single complete point cloud
    complete_cloud = np.vstack(aligned_clouds)

    # Publish the complete point cloud as a ROS message
    complete_cloud_msg = PointCloud()
    complete_cloud_msg.header = partial_clouds[
        0].header  # Use the header of the first partial cloud
    complete_cloud_msg.points = [Point32(x, y, z) for (x, y, z) in complete_cloud]
    complete_cloud_pub.publish(complete_cloud_msg)


if __name__ == '__main__':
    # Initialize the ROS node and subscribers/publishers
    rospy.init_node('cpd_registration_node')
    partial_clouds_sub = rospy.Subscriber('/partial_clouds', PointCloud, callback)
    complete_cloud_pub = rospy.Publisher('/complete_cloud', PointCloud, queue_size=10)

    # Start the ROS node
    rospy.spin()
