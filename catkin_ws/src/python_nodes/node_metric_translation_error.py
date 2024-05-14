"""This node creates translation metrics for the currently build map."""
# pylint: disable=import-error, no-name-in-module, c-extension-no-member, too-many-arguments, invalid-name, too-many-locals, too-many-instance-attributes
import csv
import math
import os
import xml.etree.ElementTree as ET
from typing import Any, Tuple

import carla
import numpy as np
import rospy
import tf
from hdl_graph_slam.msg import DynObservationArray_msg
from jsk_recognition_msgs.msg import BoundingBoxArray
from lxml import etree
from scipy.spatial.distance import cdist
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


def best_fit_transform(A, B):
    """Calculates the least-squares best-fit transform between corresponding 2D
    points in A and B."""
    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A -= centroid_A
    B -= centroid_B

    # rotation matrix
    H = np.dot(A.T, B)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    return R, t


def nearest_neighbor(src, dst):
    """Find the nearest (Euclidean) neighbor in dst for each point in src."""
    distances = cdist(src, dst, "euclidean")
    indices = np.argmin(distances, axis=1)

    return dst[indices]


def icp(A, B, max_iterations=20, tolerance=1e-8):
    """The Iterative Closest Point method."""
    src = np.copy(A)
    prev_error = 0

    for _ in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        dst = nearest_neighbor(src, B)

        # compute the transformation between the current source and nearest destination points
        R, t = best_fit_transform(src, dst)

        # update the current source
        src = (np.dot(R, src.T) + t).T

        # check error
        mean_error = np.mean(np.linalg.norm(src - dst, axis=1))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    R, t = best_fit_transform(A, src)

    return R, t, src


def get_point_from_geometry(road, s_point):
    """This function converts a road point from the opendrive xml file into a
    world coordinate."""
    for geometry in road.findall("planView/geometry"):
        start_s = float(geometry.get("s"))
        length = float(geometry.get("length"))
        end_s = start_s + length

        if start_s <= s_point <= end_s:
            x_val, y_val, hdg = (
                float(geometry.get("x")),
                float(geometry.get("y")),
                float(geometry.get("hdg")),
            )
            d_s = s_point - start_s

            if geometry.find("line") is not None:
                x_new = x_val + d_s * math.cos(hdg)
                y_new = y_val + d_s * math.sin(hdg)
                return x_new, y_new, hdg

            if geometry.find("arc") is not None:
                curvature = float(geometry.find("arc").get("curvature"))
                radius = 1 / curvature
                dhdg = d_s / radius
                theta = hdg + dhdg / 2
                x_center = x_val - radius * math.sin(hdg)
                y_center = y_val + radius * math.cos(hdg)
                x_new = x_center + radius * math.sin(theta + dhdg)
                y_new = y_center - radius * math.cos(theta + dhdg)
                return x_new, y_new, hdg + dhdg

    return None, None, None


class MetricsNode:
    """This is the node that calculated the metrics for the momentairy map.

    Those metrics include the RMSE of the keyframes to the GT path. and
    the RMSE of the discovered bounding boxes of street signs to their
    GT position.
    """

    def __init__(self):

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print(self.world)

        print("Connected to carla.")

        print(f"client: {self.client.get_client_version()}")
        print(f"server: {self.client.get_server_version()}")

        self.start_time = 0.0

        self.street_sign_positions = []
        self.keyframes_pub = rospy.Publisher("/metric_keyframes", MarkerArray, queue_size=1)
        self.icp_keyframes_pub = rospy.Publisher("/metric_keyframes_icp", MarkerArray, queue_size=1)
        self.gt_path_pub = rospy.Publisher("/metric_gt_paths", MarkerArray, queue_size=1)
        self.last_stamp = rospy.Time.now()
        self.get_street_signs_positions()

        self.filename = None

        self.listener = tf.TransformListener()
        self.gt_agent_positions = [[], [], []]

        self.rmse_street_signs = -1.0

        self.new_run_sub = rospy.Subscriber("/new_run", Header, self.new_run_callback)
        self.end_run_sub = rospy.Subscriber("/end_run", Header, self.end_run_callback)

        self.map_explored_sub = rospy.Subscriber("/map_explored", Header,
                                                 self.map_explored_callback)

        self.keyframe_sub = rospy.Subscriber(
            "/keyframes_for_metric",
            DynObservationArray_msg,
            self.optimized_keyframes_callback,
        )

        self.bbox_sub = rospy.Subscriber(
            "/feature_bounding_boxes", BoundingBoxArray,
            self.bounding_boxes_callback)  # Subscribe to /feature_bounding_boxes topic

        # Set the timer to call the callback function at a fixed rate (e.g., 10 Hz)
        rospy.Timer(rospy.Duration(0.5), lambda event: self.tf_callback())

        self.map_explored = 0.0

    def new_run_callback(self, data):
        """Callback for the new_run messages, published by the map_server."""
        print(f"New run started at {data.stamp.to_sec()}s")
        self.start_time = data.stamp.to_sec()

        self.rmse_street_signs = -1.0
        self.gt_agent_positions = [pos[-15:] for pos in self.gt_agent_positions]

        self.map_explored = 0.0

        self.filename = f"metrics_{self.start_time}.csv"

        # Writing to csv
        with open(self.filename, "a", encoding="utf-8") as file:
            # Check if file is empty
            if os.stat(self.filename).st_size == 0:
                writer = csv.writer(file)
                # Write header
                writer.writerow([
                    "timestamp",
                    "rmse0",
                    "rmse1",
                    "rmse2",
                    "rmse_of_signs",
                    "map_explored",
                ])

    def end_run_callback(self, data):
        """Callback for the end_run messages, published by the map_server."""
        print(f"End run at {data.stamp.to_sec()}")
        end_time = data.stamp.to_sec()
        run_duration = end_time - self.start_time
        print(f"Duration of the run: {run_duration}")

        completed_filename = self.filename.replace(".csv", "_completed.csv")
        os.rename(self.filename, completed_filename)
        self.filename = completed_filename  # update the filename

    def tf_callback(self):
        """Callback for the GT tf messages, published by the carla
        interface."""
        try:
            (trans, _) = self.listener.lookupTransform("world", "base_link_0", rospy.Time(0))
            self.gt_agent_positions[0].append(np.array([trans[0], trans[1]]))

            (trans, _) = self.listener.lookupTransform("world", "base_link_1", rospy.Time(0))
            self.gt_agent_positions[1].append(np.array([trans[0], trans[1]]))

            (trans, _) = self.listener.lookupTransform("world", "base_link_2", rospy.Time(0))
            self.gt_agent_positions[2].append(np.array([trans[0], trans[1]]))
        except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
        ):
            pass

    def optimized_keyframes_callback(self, data):
        """Callback for the optimized keyframe messages."""
        if self.filename is None:
            return

        self.last_stamp = data.header.stamp

        optimized_keyframes = [[], [], []]

        for marker in data.observations:
            if marker.header.seq < 3:
                optimized_keyframes[marker.header.seq].append(
                    np.array([marker.pose.position.x, marker.pose.position.y]))

        print()
        rmses = []
        for index, keyframes in enumerate(optimized_keyframes):
            self.visualize_pos_list(
                keyframes,
                publisher=self.keyframes_pub,
                color=(1.0, 1.0, 1.0 * (index / len(optimized_keyframes))),
                idx=index,
            )
            self.visualize_pos_list(
                self.gt_agent_positions[index],
                publisher=self.gt_path_pub,
                color=(0.5, 1.0 * (index / len(optimized_keyframes)), 0.5),
                size=0.7,
                idx=index,
            )

            rmse = self.calculate_rmse(estimations=keyframes,
                                       ground_truth=self.gt_agent_positions[index])
            print(f"Agent {index}: RMSE for {len(keyframes):4} keyframes: \t{rmse:6}m")
            rmses.append(rmse)

        timestamp = data.header.stamp.to_sec() - self.start_time

        if timestamp > 0:
            # Writing to csv
            with open(self.filename, "a", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([round(timestamp, 2)] + rmses + [self.rmse_street_signs] +
                                [self.map_explored])

        self.keyframe_sub.unregister()
        self.keyframe_sub = rospy.Subscriber(
            "/keyframes_for_metric",
            DynObservationArray_msg,
            self.optimized_keyframes_callback,
        )

    def calculate_rmse(self, estimations, ground_truth):
        """This function calculates the rmse given the estimations and the
        ground truth positions."""
        if not estimations or not ground_truth:
            return -1.0

        squared_errors = []
        for estimation in estimations:
            min_distance = np.inf
            for i in range(len(ground_truth) - 1):
                point_a = np.array(ground_truth[i])
                point_b = np.array(ground_truth[i + 1])
                vec_ab = point_b - point_a
                vec_ac = estimation - point_a
                val_t = np.dot(vec_ac, vec_ab) / np.dot(vec_ab, vec_ab)
                if 0 <= val_t <= 1:
                    point_d = point_a + val_t * vec_ab
                    distance = np.linalg.norm(estimation - point_d)
                else:
                    distance = min(
                        np.linalg.norm(estimation - point_a),
                        np.linalg.norm(estimation - point_b),
                    )
                if distance < min_distance:
                    min_distance = distance
            squared_errors.append(min_distance**2)
        rmse = np.sqrt(np.mean(squared_errors))

        return round(rmse, 3)

    def map_explored_callback(self, data):
        """This callback receives the map_explored messages, sent by the map
        server."""
        self.map_explored = data.seq / 10000.0
        print(f"Map explored: {self.map_explored * 100} %")

    def bounding_boxes_callback(self, data):
        """This is the callback for the bounding boxes."""
        if self.filename is None:
            return

        self.last_stamp = data.header.stamp

        # Assuming data.bounding_boxes is an array of BoundingBoxWithLabel
        label_1_boxes_positions = [
            np.array([box.pose.position.x, box.pose.position.y]) for box in data.boxes
            if box.label == 1
        ]
        rmse = self.calculate_rmse(estimations=label_1_boxes_positions,
                                   ground_truth=self.street_sign_positions)
        print(f"RMSE for {len(label_1_boxes_positions)} street-signs: {rmse}")
        self.rmse_street_signs = rmse

        self.bbox_sub.unregister()
        self.bbox_sub = rospy.Subscriber(
            "/feature_bounding_boxes", BoundingBoxArray,
            self.bounding_boxes_callback)  # Subscribe to /feature_bounding_boxes topic

    def get_street_signs_positions(self):
        """This function receives the roadnetwork from the carla server, and
        takes the ground truth position of the street signs from this data."""
        road_network = None
        while not road_network:
            road_network = self.world.get_map()

        self.street_sign_positions = self.get_traffic_sign_positions(road_network)
        print(f"Traffic sign positions: {len(self.street_sign_positions)}")

    def get_traffic_sign_positions(self, road_network):
        """This is the function to parse the sign objects from the opendrive
        xml file, that the carla server delivers."""
        opendrive_xml = road_network.to_opendrive()
        opendrive_xml = opendrive_xml.split("\n", 1)[-1]

        opendrive_root = etree.fromstring(opendrive_xml)

        # Find all street signs and their associated roads
        sign_positions = []
        for obj in opendrive_root.iter("object"):

            obj_name = obj.get("name")
            if obj_name and ("Speed" in obj_name or "speed" in obj_name
                             ):  # Adjust this condition to match other types of street signs
                s_value = float(obj.get("s"))
                t_value = float(obj.get("t"))
                road_id = obj.getparent().getparent().get("id")

                # Find the corresponding road
                road = opendrive_root.find(f".//road[@id='{road_id}']")

                # Get the world coordinates (x, y) and heading (hdg) along the reference line
                x_ref, y_ref, hdg = get_point_from_geometry(road, s_value)

                # Calculate the world coordinates of the street sign
                if x_ref is not None and y_ref is not None and hdg is not None:
                    x_sign = x_ref + t_value * math.cos(hdg + math.pi / 2)
                    y_sign = y_ref + t_value * math.sin(hdg + math.pi / 2)
                    sign_positions.append((x_sign, y_sign))

                    self.world.debug.draw_string(
                        carla.Location(x_sign, y_sign),
                        "O",
                        draw_shadow=False,
                        color=carla.Color(r=0, g=255, b=0),
                        life_time=20.0,
                        persistent_lines=True,
                    )

        return sign_positions

    def pretty_print_tree(self, tree):
        """This helper function prints the xml tree of the received road
        network."""
        pretty_tree_str = ET.tostring(tree.getroot(), encoding="utf-8",
                                      method="xml").decode("utf-8")
        print(pretty_tree_str)

    def visualize_pos_list(
        self,
        pos_list: Any,
        publisher: Any,
        color: Tuple[float, float, float] = (1, 1, 1),
        height_offset: float = 0.0,
        alpha: float = 1.0,
        size: float = 1.0,
        idx: int = None,
    ) -> None:
        """This function publishes an array of small bounding boxes, that
        represents the given observations."""

        # Create MarkerArray message
        marker_array = MarkerArray()

        # Create Marker messages for nodes
        for index, pos in enumerate(pos_list):
            node_marker = Marker()
            node_marker.header.frame_id = "world"
            node_marker.header.stamp = self.last_stamp
            node_marker.type = Marker.CYLINDER
            node_marker.ns = "nodes"
            node_marker.id = idx * 100000 + index if idx is not None else index
            node_marker.pose.position.x = pos[0]
            node_marker.pose.position.y = pos[1]
            node_marker.pose.position.z = 2.0 + height_offset
            node_marker.pose.orientation.w = 1
            node_marker.pose.orientation.x = 0
            node_marker.pose.orientation.y = 0
            node_marker.pose.orientation.z = 0
            node_marker.frame_locked = True
            node_marker.scale.x = size
            node_marker.scale.y = size
            node_marker.scale.z = size * 2
            node_marker.color.r = color[0]
            node_marker.color.g = color[1]
            node_marker.color.b = color[2]
            node_marker.color.a = alpha
            marker_array.markers.append(node_marker)

        publisher.publish(marker_array)

    def spin(self) -> None:
        """This in not really needed but to keep the node alive."""


def main() -> None:
    """Init the node, and start it."""
    rospy.init_node("metrics_node")
    node = MetricsNode()

    print("Init done.")

    # pylint: disable=duplicate-code
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()


if __name__ == "__main__":
    main()
