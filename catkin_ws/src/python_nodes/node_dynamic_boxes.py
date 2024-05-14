"""This is a helper node, to visualize the detected street signs of the agents.

this module is a node, that reads the traffic sign data from the
semantic velodyne points and generates bounding boxes around the traffic
signs by using a DBScan clustering algorithm.
"""
# pylint: disable=duplicate-code, import-error, too-few-public-methods, too-many-locals
import time
from typing import Any, List, Tuple

import numpy as np
import rospy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


class DynamicObject:
    """Data class to hold the id and the list of points belonging to this
    id."""

    object_id: int
    frames: List[Tuple[float, ...]] = []

    def __init__(self, object_id: int, frame: Tuple[float, ...]) -> None:
        self.object_id = object_id
        self.frames = []
        self.frames.append(frame)

    def add_point(self, point: Tuple[float, ...]) -> None:
        """Adds another point to the list."""
        self.frames.append(point)


class NodeDynamicBoxes:
    """This class is the node that clusters the traffic sign point clouds and
    generates the bounding boxes for all signs."""

    def __init__(self) -> None:

        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_dyn_0", PointCloud2,
                                            self.map_points_callback, 0)

        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_dyn_1", PointCloud2,
                                            self.map_points_callback, 1)

        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_dyn_2", PointCloud2,
                                            self.map_points_callback, 2)

    @staticmethod
    def print_array(data: List[Any]) -> None:
        """Helper function to get a fast output of the content."""
        for i, obj in enumerate(data):
            print(f"{i} : {obj.object_id} : {len(obj.frames)}")

    @staticmethod
    def process_data(data: pc2, agent_no: int) -> None:
        """Outsourced method."""

        pcl_poles: List[DynamicObject] = []

        for point in pc2.read_points(data, skip_nans=True):
            added = False
            for obj in pcl_poles:
                if int(point[4]) == obj.object_id:
                    obj.add_point(point)
                    added = True

            if not added:
                item = DynamicObject(int(point[4]), point)
                pcl_poles.append(item)

        pub = rospy.Publisher(f"/dyn_boxes_{agent_no}", BoundingBoxArray, queue_size=10)
        boxes: BoundingBoxArray = BoundingBoxArray()
        boxes.boxes = []

        for obj in pcl_poles:
            if obj.object_id > 0:
                boxes.boxes.append(create_bounding_box(obj.frames, data.header, obj.object_id))

        boxes.header.frame_id = f"base_link_{agent_no}"
        boxes.header.stamp = data.header.stamp
        pub.publish(boxes)

    def map_points_callback(self, data: pc2, agent_no: int) -> None:
        """This is the callback that gets the data and the index of the agent
        processes the data and then republishes again."""
        print()
        print(f"Agent {agent_no} - New clouds received: {time.strftime('%H:%M:%S')}")

        start_time = time.time()

        self.process_data(data, agent_no)

        print(f"Agent {agent_no} - Calculation time: {((time.time() - start_time) * 1000):.0f}ms")

    def spin(self) -> None:
        """This in not really needed but to keep the node alive."""


def create_bounding_box(pcl_poles: PointCloud2, header: Any, object_id: int) -> BoundingBox:
    """Returns a bounding box element that can be appended to a
    boundingboxarray."""
    box: BoundingBox = BoundingBox()
    box.header.stamp = header.stamp
    box.header.frame_id = header.frame_id
    box.pose.orientation.w = 1

    box.value = object_id
    box.label = object_id

    pcl = np.asarray(pcl_poles)

    vec_min: Tuple[float, float, float] = np.min(pcl, axis=0)
    vec_max: Tuple[float, float, float] = np.max(pcl, axis=0)

    padding: float = 1.0

    x_min: float = vec_min[0] - padding
    x_max: float = vec_max[0] + padding
    y_min: float = vec_min[1] - padding
    y_max: float = vec_max[1] + padding
    z_min: float = vec_min[2] - padding
    z_max: float = vec_max[2] + padding

    x_pos: float = (x_max - x_min) / 2 + x_min
    y_pos: float = (y_max - y_min) / 2 + y_min
    # z_pos: float = (z_max - z_min) / 2 + z_min
    z_pos: float = 0.0

    box.pose.position.x = x_pos
    box.pose.position.y = y_pos
    box.pose.position.z = z_pos

    box.dimensions.x = x_max - x_min
    # box.dimensions.x = 3.0
    box.dimensions.y = y_max - y_min
    # box.dimensions.y = 3.0
    box.dimensions.z = z_max - z_min
    # box.dimensions.z = 5.0

    # print(
    #     f"x: {x_pos}, y: {y_pos}, z: {z_pos} dimx: {box.dimensions.x}, dimy: {box.dimensions.y}"
    # )

    return box


def main() -> None:
    """Init the node, and start it."""
    rospy.init_node("cluster_node")
    node = NodeDynamicBoxes()

    # pylint: disable=duplicate-code
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()


if __name__ == "__main__":
    main()
