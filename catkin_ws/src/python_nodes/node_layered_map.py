"""This ros node will process the incoming pointclouds from the sensor and
simulate a semantic network that could differentiate between different classes.
the input are the pointclouds from the map server, the output is a layered
point cloud where the different semantic classes have different height. The
output of this node is used for the clustering of the poles and street signs.

The definition of the separation is contained withing a xml file
"""
# pylint: disable=too-few-public-methods, import-error
from typing import List, Tuple

import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


class GraphLevels:
    """RGB Colors definition."""

    LEVEL_1 = 1000
    LEVEL_2 = 1000
    LEVEL_3 = 75
    LEVEL_4 = 75


class NodeLayeredMap:
    """This class is the node that is dividing the pointclouds."""

    def __init__(self) -> None:

        self.map_graph: List[Tuple[float, ...]] = []
        self.sub_agent_0 = rospy.Subscriber("/map_server/map_points", PointCloud2,
                                            self.map_points_callback)

        # yapf: disable
        self.class_height_mod: List[Tuple[bool, int]] = [
            (False, 1500),  # None
            (False, 1000),  # Building
            (True, GraphLevels.LEVEL_2),  # Fences
            (False, 1200),  # Other
            (False, 1000),  # Pedestrians
            (True, GraphLevels.LEVEL_3),  # Poles
            (True, GraphLevels.LEVEL_1),  # Roadlines
            (True, GraphLevels.LEVEL_1),  # Roads
            (False, 1000),  # Sidewalks
            (False, 1000),  # Vegetation
            (False, 1000),  # Vehicles
            (True, GraphLevels.LEVEL_2),  # Walls
            (True, GraphLevels.LEVEL_4),  # TrafficSigns
        ]
        # yapf: enable

    def process_data(self, data: PointCloud2) -> None:
        """Outsourced method."""

        for point in pc2.read_points(data, skip_nans=True):
            # Adding the custom value on top of the Z value to shift the different
            # semantic classes to the top and stack them
            point = (
                point[0],
                point[1],
                point[2] + self.class_height_mod[int(point[6])][1] if point[6] <= 12 else point[2] +
                20000,
                0,
                0,
                0,
                point[6],
                point[7],
            )

            if point[6] <= 12:
                if self.class_height_mod[int(point[6])][0]:
                    self.map_graph.append(point)

        pub = rospy.Publisher("/graph", PointCloud2, queue_size=10)

        graph_msg: PointCloud2 = pc2.create_cloud(data.header, data.fields, self.map_graph)

        pub.publish(graph_msg)

    def map_points_callback(self, data: PointCloud2) -> None:
        """This is the callback that gets the data and the index of the agent
        processes the data and then republishes again."""
        print(f"New map received at time: {data.header.stamp.to_sec():.2f}s")

        # reset the old graph and start a new one
        self.map_graph = []
        self.process_data(data)

    def spin(self) -> None:
        """This in not really needed but to keep the node alive."""


def main() -> None:
    """Init the node, and start it."""
    rospy.init_node("graph_builder")
    node = NodeLayeredMap()

    print("Init done.")

    # pylint: disable=duplicate-code
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()


if __name__ == "__main__":
    main()
