"""This is the node, that creates the metrics for the different intersection
detection approaches."""
# pylint: disable=line-too-long, import-error, unspecified-encoding, consider-using-f-string, too-many-instance-attributes
import datetime
import os
from typing import List, Any

import numpy as np
import rospy
from rosgraph_msgs.msg import Clock
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray


def save_ply_file(points, filename):
    """Function to save the ply file, where the intersection points are
    stored."""
    with open(filename, "w") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("element vertex {}\n".format(len(points)))
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")

        for point in points:
            file.write("{} {} {}\n".format(point[0], point[1], point[2]))


class IntersectionMetricNode:
    """This is the node to collect all the metrics from the different
    approaches for the intersection detection."""

    def __init__(self) -> None:

        self.base_path = "."
        self.town_name = "town02"

        self.running = False

        self.start_time = 0.0

        self.folder_name = f"{self.town_name}_{datetime.datetime.now().strftime('%B-%d_%I-%M')}_starting"

        # Create 'morphology' subfolder
        self.morphology_path = os.path.join(self.base_path, self.folder_name, "morphology")

        # Create 'lanegraph' subfolder
        self.lanegraph_path = os.path.join(self.base_path, self.folder_name, "lanegraph")

        # Global variables to store marker positions
        self.intersection_nodes_lanegraph: List[Any] = []
        self.intersection_nodes_morphology: List[Any] = []
        self.lanegraph_intersection_nodes: List[Any] = []
        self.street_points: List[Any] = []
        self.clock_ticks = 0

        rospy.Subscriber(
            "/intersection_street_nodes_morphology",
            MarkerArray,
            self.intersection_nodes_morphology_callback,
        )
        rospy.Subscriber(
            "/intersection_street_nodes",
            MarkerArray,
            self.intersection_nodes_lanegraph_callback,
        )
        rospy.Subscriber("/map_server/map_points", PointCloud2, self.map_points_callback)
        rospy.Subscriber(
            "lane_graph_intersection_nodes",
            MarkerArray,
            self.lane_graph_intersection_node_callback,
        )

        rospy.Subscriber("/end_run", Header, self.end_run_callback)
        rospy.Subscriber("/new_run", Header, self.new_run_callback)

        rospy.Subscriber("/clock", Clock, self.clock_callback)

    def spin(self) -> None:
        """This in not really needed but to keep the node alive."""

    def create_paths(self):
        """This method creates the folders to put the final metric files in."""
        if not os.path.exists(self.morphology_path):
            os.makedirs(self.morphology_path)
            print("Created folder: morphology")

        if not os.path.exists(self.lanegraph_path):
            os.makedirs(self.lanegraph_path)
            print("Created folder: lanegraph")

    def new_run_callback(self, data):
        """Callback for the new_run messages, published by the map_server."""
        print(f"New run started at {data.stamp.to_sec()}s")
        self.start_time = data.stamp.to_sec()

        self.folder_name = f"{self.town_name}_{datetime.datetime.now().strftime('%B-%d_%I-%M')}_{int(self.start_time)}"

        # Create 'morphology' subfolder
        self.morphology_path = os.path.join(self.base_path, self.folder_name, "morphology")

        # Create 'lanegraph' subfolder
        self.lanegraph_path = os.path.join(self.base_path, self.folder_name, "lanegraph")

        self.create_paths()

        print(f"Now saving to: {self.folder_name}")
        self.running = True

    def end_run_callback(self, data):
        """Callback for the end_run messages, published by the map_server."""
        print("Run ended.")
        self.running = False

        print(f"End run at {data.stamp.to_sec()}")
        end_time = data.stamp.to_sec()
        run_duration = end_time - self.start_time
        print(f"Duration of the run: {run_duration}")

        self.clock_ticks = 0
        self.intersection_nodes_lanegraph = []
        self.intersection_nodes_morphology = []
        self.lanegraph_intersection_nodes = []
        self.street_points = []

    def map_points_callback(self, data):
        """Callback for the map point messages, published by the map_server."""
        self.street_points = []

        # Iterate over the PointCloud2 data
        for point in pc2.read_points(data, skip_nans=True):
            point_x, point_y, point_z = point[:3]
            semantic_class = point[6]

            # Filter points with semantic class 7
            if semantic_class == 7:
                self.street_points.append([point_x, point_y, point_z])

    # Callback functions to save positions
    def intersection_nodes_lanegraph_callback(self, data):
        """Callback for the intersection messages, published by the lane graph
        node."""
        self.intersection_nodes_lanegraph = [(marker.pose.position.x, marker.pose.position.y,
                                              marker.pose.position.z) for marker in data.markers]

    def intersection_nodes_morphology_callback(self, data):
        """Callback for the intersection messages, published by the morphology
        node."""
        self.intersection_nodes_morphology = [(marker.pose.position.x, marker.pose.position.y,
                                               marker.pose.position.z) for marker in data.markers]

    def lane_graph_intersection_node_callback(self, data):
        """Callback for the lane graph messages, published by the lane graph
        node."""
        self.lanegraph_intersection_nodes = [(marker.pose.position.x, marker.pose.position.y,
                                              marker.pose.position.z) for marker in data.markers]

    def clock_callback(self, _):
        """Callback for the clock messages."""
        if not self.running:
            return

        self.clock_ticks += 1
        if self.clock_ticks % 10 == 0:

            new_lanegraph_path = os.path.join(self.lanegraph_path,
                                              str(int(self.clock_ticks / 10)).zfill(4))
            new_morphology_path = os.path.join(self.morphology_path,
                                               str(int(self.clock_ticks / 10)).zfill(4))

            if not os.path.exists(new_lanegraph_path):
                os.makedirs(new_lanegraph_path)

            if not os.path.exists(new_morphology_path):
                os.makedirs(new_morphology_path)

            np.savez(
                os.path.join(new_lanegraph_path, "intersections.npz"),
                np.array(self.intersection_nodes_lanegraph),
            )
            np.savez(
                os.path.join(new_lanegraph_path, "intersection_nodes.npz"),
                np.array(self.lanegraph_intersection_nodes),
            )
            np.savez(
                os.path.join(new_morphology_path, "intersections.npz"),
                np.array(self.intersection_nodes_morphology),
            )

            # Convert to NumPy array and save as .ply file
            points_array = np.array(self.street_points, dtype=np.float32)
            save_ply_file(points_array, os.path.join(new_morphology_path, "street_cloud.ply"))

            print(f"Files written for step: {str(int(self.clock_ticks / 10)).zfill(4)} - "
                  f"Lanegraph: {len(self.intersection_nodes_lanegraph)} "
                  f"Nodes: {len(self.lanegraph_intersection_nodes)} "
                  f"Morphology: {len(self.intersection_nodes_morphology)} "
                  f"Street Points: {len(self.street_points)}")


# Initialize ROS node
def main():
    """Main fuction."""
    rospy.init_node("intersection_metric_node", anonymous=True)

    node = IntersectionMetricNode()

    print("Init done.")

    rate = rospy.Rate(2.0)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()


if __name__ == "__main__":
    main()
