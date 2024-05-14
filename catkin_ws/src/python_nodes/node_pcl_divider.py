"""
LEGACY - this function is now done as cpp node (much faster).
- - - - - - - - - - - - - - - - - - 
this ros node will process the incoming pointclouds from the sensor
and simulate a semantic network that could differentiate between different classes.
the input are the pointclouds from the agents,
the output are 2 pointclouds per agent:
    - one containing static object
    - one containing dynamic objects.
"""
# pylint: skip-file
import sys
import time
from typing import List, Tuple

import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


def process_data(data: pc2, agent_no: int) -> Tuple[PointCloud2, PointCloud2]:
    """
    outsourced method
    """

    static_objects: List[Tuple[float, ...]] = []
    dynamic_objects: List[Tuple[float, ...]] = []

    for point in pc2.read_points(data, skip_nans=True):
        if int(point[3]) == 4 or (point[3]) == 10:
            dynamic_objects.append(point)
        else:
            static_objects.append(point)

    msg_static: PointCloud2 = pc2.create_cloud(data.header, data.fields, static_objects)

    header_dyn = data.header
    header_dyn.frame_id = f"base_link_{agent_no}"
    msg_dyn: PointCloud2 = pc2.create_cloud(header_dyn, data.fields, dynamic_objects)

    return msg_dyn, msg_static


def publish_data(msg_dyn: PointCloud2, msg_static: PointCloud2, agent_no: int) -> None:
    """
    outsourced method to keep thing separated
    """
    pub = rospy.Publisher(f"/velodyne_points_stat_{agent_no}", PointCloud2, queue_size=10)
    pub.publish(msg_static)

    pub = rospy.Publisher(f"/velodyne_points_dyn_{agent_no}", PointCloud2, queue_size=10)
    pub.publish(msg_dyn)


class NodePointCloudDivider:
    """
    this class is the node that is dividing the pointclouds
    """

    def __init__(self) -> None:
        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_raw_0", PointCloud2,
                                            self.raw_pcl_callback, 0)
        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_raw_1", PointCloud2,
                                            self.raw_pcl_callback, 1)
        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_raw_2", PointCloud2,
                                            self.raw_pcl_callback, 2)
        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_raw_3", PointCloud2,
                                            self.raw_pcl_callback, 3)

    @staticmethod
    def raw_pcl_callback(data: PointCloud2, agent_no: int) -> None:
        """
        this is the callback that gets the data and the index of the agent
        processes the data and then republishes again
        """
        start = time.time()
        msg_dyn: PointCloud2
        msg_static: PointCloud2
        msg_dyn, msg_static = process_data(data, agent_no)
        publish_data(msg_dyn, msg_static, agent_no=agent_no)
        print(f"Agent {agent_no} - Computation Time: {(time.time() - start):.3f}")

    def spin(self) -> None:
        """
        this in not really needed but to keep the node alive
        """


def main() -> None:
    """
    init the node, and start it
    """

    print()
    print("This node is deprecated. Use the C++ Version. ")
    print("Run it with: rosrun hdl_graph_slam pcl_divider_node")
    print()

    sys.exit()

    rospy.init_node("point_cloud_divider")
    node = NodePointCloudDivider()

    # pylint: disable=duplicate-code
    # rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        # rate.sleep()
        time.sleep(1.0)


if __name__ == "__main__":
    main()
