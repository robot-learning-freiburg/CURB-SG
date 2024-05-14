"""Helper function to check for differnt properties of point clouds."""
# pylint: skip-file

import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


def process_data(data):
    """Prints the date from the chosen topic."""
    ind = 0
    # max_class = 0.0

    id_dict = {}

    for point in pc2.read_points(data, skip_nans=True):
        if point[4] > 0 and point[4] not in id_dict and point[3] in (4, 10):
            ind += 1
            id_dict[point[4]] = point[3]

    print(id_dict)
    print(f"{ind} = {len(id_dict)}")
    # print(max_class)


class NodePointCloudAnalyzer:
    """This class is the node that is dividing the pointclouds."""

    def __init__(self):
        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_raw_0", PointCloud2,
                                            self.pcl_callback)

        # self.sub_agent_0 = rospy.Subscriber(
        #     "/velodyne_points_stat_0", PointCloud2, self.pcl_callback
        # )

    @staticmethod
    def pcl_callback(data):
        """This is the callback that gets the data and the index of the agent
        processes the data and then republishes again."""
        print("-----------")
        process_data(data)
        print("-----------\n\n")

    def spin(self):
        """This in not really needed but to keep the node alive."""


def main():
    """Init the node, and start it."""
    rospy.init_node("point_cloud_analyzer")
    node = NodePointCloudAnalyzer()

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()


if __name__ == "__main__":
    main()
