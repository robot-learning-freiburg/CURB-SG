"""This node will check the current observations for the latest entry.

if this entry is not older than a threshold and close to a node of the
lanegraph, the observation will be displayed on the node of the
lanegraph.
"""
# pylint: disable=import-error, no-name-in-module, too-many-locals, too-many-statements
import math
import numpy as np
import rospy
from hdl_graph_slam.msg import DynObservationArray_msg
from visualization_msgs.msg import Marker, MarkerArray
import networkx as nx


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler Angles to Quaternion :param roll: Roll angle in radians
    :param pitch: Pitch angle in radians :param yaw: Yaw angle in radians
    :return: Quaternion as a list [x, y, z, w]"""
    quaternion_x = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(
        roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    quaternion_y = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(
        roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
    quaternion_z = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(
        roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
    quaternion_w = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(
        roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)

    return [quaternion_x, quaternion_y, quaternion_z, quaternion_w]


def delete_all_displayed_markerarrays(pub):
    """Function to delete all displayed markerarrays on a given publisher."""
    marker = Marker()
    marker.action = Marker.DELETEALL
    delete_all_marker_array = MarkerArray()
    delete_all_marker_array.markers.append(marker)
    pub.publish(delete_all_marker_array)


def marker_array_to_nx_graph(marker_array: MarkerArray):
    """This function converts a marker array to a nxdigraph."""
    graph = nx.DiGraph()

    for marker in marker_array.markers:
        node_id = marker.id
        pos = (marker.pose.position.x, marker.pose.position.y)

        if marker.type == Marker.SPHERE:
            graph.add_node(node_id, pos=pos)
        elif marker.type == Marker.ARROW:
            if marker.text != "":
                # print(marker.text)
                start_id, end_id = map(int, marker.text.split(";"))
                graph.add_edge(start_id, end_id)

    return graph


class VehiclesOnLanegraphNode:
    """This is the node for the functionality described in the file
    docstring."""

    def __init__(self):
        self.lane_graph_height_offset = 40
        self.lane_graph_markers = []
        self.lane_graph_reconstructed = nx.DiGraph()
        self.last_stamp = None
        rospy.Subscriber("/lane_graph", MarkerArray, self.lane_graph_callback)
        rospy.Subscriber(
            "/map_server/observation_array",
            DynObservationArray_msg,
            self.observations_callback,
        )

        self.current_vehicle_positions_pub = rospy.Publisher("/current_vehicle_positions",
                                                             MarkerArray,
                                                             queue_size=10)

    def get_orientation_from_graph(self, current_node):
        """This function returns the orientation of a given node within the
        graph."""
        if not self.lane_graph_reconstructed.in_edges(current_node):
            return None

        # Assuming the first outgoing edge is what we are interested in.
        prev_node = list(self.lane_graph_reconstructed.in_edges(current_node))[0][0]

        current_pos = np.array(self.lane_graph_reconstructed.nodes[current_node]["pos"])
        prev_pos = np.array(self.lane_graph_reconstructed.nodes[prev_node]["pos"])

        direction_vector = current_pos - prev_pos

        # Calculate the angle using arctan2
        angle = math.atan2(direction_vector[1], direction_vector[0])

        return angle

    def lane_graph_callback(self, marker_array):
        """Callback for the lane grpah markers."""
        self.lane_graph_markers = marker_array.markers

        self.lane_graph_reconstructed = marker_array_to_nx_graph(marker_array)

    def observations_callback(self, observations_array):
        """Callback for the current observations."""
        if not self.lane_graph_markers:
            return

        print()
        print(f"""number of nodes: {len(self.lane_graph_reconstructed.nodes)}
                edges: {len(self.lane_graph_reconstructed.edges)}""")

        self.last_stamp = observations_array.header.stamp
        newest_vehicles = {}

        for observation in observations_array.observations:
            vehicle_id = observation.header.seq

            timestamp_difference = (observations_array.header.stamp - observation.header.stamp)

            if timestamp_difference <= rospy.Duration(30):
                if (vehicle_id not in newest_vehicles
                        or observation.header.stamp > newest_vehicles[vehicle_id][0].header.stamp):

                    newest_vehicles[vehicle_id] = (observation, timestamp_difference)

        current_vehicle_positions = MarkerArray()

        for vehicle_id, vehicle_tuple in newest_vehicles.items():
            vehicle_observation = vehicle_tuple[0]
            min_distance = float("inf")
            closest_sphere = None
            vehicle_position = np.array([
                vehicle_observation.pose.position.x,
                vehicle_observation.pose.position.y,
            ])

            for marker in self.lane_graph_markers:
                if marker.type == Marker.SPHERE:
                    sphere_position = np.array([marker.pose.position.x, marker.pose.position.y])
                    distance = np.linalg.norm(vehicle_position - sphere_position)

                    if distance < min_distance:
                        min_distance = distance
                        closest_sphere = marker

            if min_distance < 1.5 or min_distance < 2.0:
                size = 3.0
                new_sphere = Marker()
                new_sphere.header.frame_id = "world"
                new_sphere.header.stamp = self.last_stamp
                new_sphere.id = vehicle_id
                new_sphere.type = Marker.CUBE
                vehicle_pose = closest_sphere.pose
                vehicle_pose.position.z = self.lane_graph_height_offset + 6.0
                new_sphere.pose = vehicle_pose
                new_sphere.scale.x = size * 1.8
                new_sphere.scale.y = size
                new_sphere.scale.z = size / 1.8
                new_sphere.color.a = 1.0
                new_sphere.color.r = 1.0
                new_sphere.color.g = 0.0
                new_sphere.color.b = 0.0
                new_sphere.ns = "current_vehicle_positions"

                angle = self.get_orientation_from_graph(closest_sphere.id)  # G is your nx.DiGraph

                if angle is not None:
                    quaternion = euler_to_quaternion(0, 0, angle)

                    new_cube = Marker()
                    new_cube.type = Marker.CUBE
                    new_cube.pose = closest_sphere.pose
                    new_cube.pose.orientation.x = quaternion[0]
                    new_cube.pose.orientation.y = quaternion[1]
                    new_cube.pose.orientation.z = quaternion[2]
                    new_cube.pose.orientation.w = quaternion[3]

                current_vehicle_positions.markers.append(new_sphere)
        print(f"Found {len(current_vehicle_positions.markers)} vehicles on lanegraph.")

        self.current_vehicle_positions_pub.publish(current_vehicle_positions)

    def spin(self) -> None:
        """This in not really needed but to keep the node alive."""


def main() -> None:
    """Init the node, and start it."""
    rospy.init_node("vehicle_position_visualizer")
    node = VehiclesOnLanegraphNode()

    print("Init done.")

    # pylint: disable=duplicate-code
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()


if __name__ == "__main__":
    main()
