"""This is the ros node, that creates the root node of the scene graph and also
the edges from this node to the intersection/street edges and to the bounding
boxes of the feature (poles/signs)"""
# pylint: disable=import-error
import math
import random

import numpy as np
import rospy
from jsk_recognition_msgs.msg import BoundingBoxArray
from shapely.geometry import LineString
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def euclidean_distance(point1, point2):
    """Helper function."""
    x_diff = point1.x - point2.x
    y_diff = point1.y - point2.y
    z_diff = point1.z - point2.z
    distance = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
    return distance


def point_to_numpy(point):
    """Convert point to numpy array."""
    return np.array([point.x, point.y])


def bounding_box_to_polygon(box):
    """Converts a bounding box object into a Shapely Polygon."""
    box_center = point_to_numpy(box.pose.position)
    box_size = point_to_numpy(box.dimensions) + 4.0
    box_min = box_center - box_size / 2
    box_max = box_center + box_size / 2

    return Polygon([
        (box_min[0], box_min[1]),
        (box_max[0], box_min[1]),
        (box_max[0], box_max[1]),
        (box_min[0], box_max[1]),
    ])


def nearest_bounding_box_shapely(point, bounding_boxes):
    """Retruns the neares bounding box to the given point."""
    min_distance = float("inf")
    nearest_box = None

    shapely_point = ShapelyPoint(point.x, point.y)

    for box in bounding_boxes:
        box_center = box.pose.position
        shapely_center = ShapelyPoint(box_center.x, box_center.y)

        polygon = bounding_box_to_polygon(box)

        # Check if the point is inside the polygon
        if polygon.contains(shapely_point):
            nearest_box = box
            break

        line = LineString([shapely_point, shapely_center])
        intersection = polygon.intersection(line)
        distance = shapely_point.distance(intersection)

        if min_distance > distance > 0:
            min_distance = distance
            nearest_box = box

    return nearest_box


class IntersectionKeyframeNode:
    """This is the class of the node, that subscribes to the specific topics
    where the data is published that is relevant for the scene graph
    generation."""

    def __init__(self):
        self.keyframes = []
        self.intersection_nodes = []
        self.bounding_boxes = []
        self.feature_bounding_boxes = []

        self.central_node_height_offset = 30
        self.central_node_size = 6.0

        rospy.Subscriber("/map_server/markers", MarkerArray, self.markerarray_callback)
        rospy.Subscriber("/intersection_street_nodes", MarkerArray, self.intersection_callback)
        rospy.Subscriber("/node_bounding_boxes", BoundingBoxArray, self.bounding_box_callback)
        rospy.Subscriber(
            "/feature_bounding_boxes",
            BoundingBoxArray,
            self.feature_bounding_box_callback,
        )

        self.pub = rospy.Publisher("/intersection_keyframe_edges", MarkerArray, queue_size=10)

    def spin(self) -> None:
        """This in not really needed but to keep the node alive."""

    def markerarray_callback(self, data):
        """Callback function for the marker array."""
        for marker in data.markers:
            if marker.ns == "nodes":
                self.keyframes = []

                for keyframe_position in marker.points:
                    self.keyframes.append(keyframe_position)

    def intersection_callback(self, data):
        """Callback function for the intersection nodes."""
        print("Nodes inc")
        self.intersection_nodes = data.markers
        print(f"Received nodes: {len(self.intersection_nodes)}")

    def bounding_box_callback(self, data):
        """Callback function for the bounding boxes."""
        print("Bbox inc")
        self.bounding_boxes = data.boxes
        print(f"Received boxes: {len(self.bounding_boxes)}")
        self.process_bounding_boxes()

    def feature_bounding_box_callback(self, data):
        """Callback function for the feature bouding boxes."""
        print("Feature Bbox inc")

        self.feature_bounding_boxes = [box.pose.position for box in data.boxes]

    def process_bounding_boxes(self):
        """This function publishes the edges that are created based on the
        received boudning boxes."""
        # Order the bounding boxes by their area (smallest first)
        sorted_bounding_boxes = sorted(self.bounding_boxes,
                                       key=lambda box: box.dimensions.x * box.dimensions.y)

        marker_array = MarkerArray()

        for box in sorted_bounding_boxes:
            created_edges = 0

            for feature in list(self.feature_bounding_boxes):
                if self.is_within_bounding_box(feature, box, padding=2.0):
                    created_edges += 1
                    line_marker = self.create_line_marker(
                        feature,
                        box,
                    )
                    #   color=ColorRGBA(0.2,0.2,0.2, 1.0))
                    if line_marker is not None:
                        marker_array.markers.append(line_marker)
                        self.feature_bounding_boxes.remove(feature)

            # as the features can easily be outside the bounding boxes, as
            # the bounding boxes are created from the lane graph, and features
            # don't necessarily have to be close to the roads
            # we now assign those remaining features to the closest nodes
            print(f"Boxes: {len(self.feature_bounding_boxes)}")
            if len(self.feature_bounding_boxes) > 0:
                for feature in self.feature_bounding_boxes:
                    nearest_box = nearest_bounding_box_shapely(feature, sorted_bounding_boxes)

                    if nearest_box is not None:
                        created_edges += 1
                        line_marker = self.create_line_marker(
                            feature,
                            nearest_box,
                        )
                        if line_marker is not None:
                            marker_array.markers.append(line_marker)

                    else:
                        print("box was none")

        (
            central_node_marker,
            central_node_connections,
        ) = self.create_central_node_and_connections()
        marker_array.markers.append(central_node_marker)
        for connection in central_node_connections:
            marker_array.markers.append(connection)

        if marker_array is not None:
            marker = Marker()
            marker.action = Marker.DELETEALL
            delete_all_marker_array = MarkerArray()
            delete_all_marker_array.markers.append(marker)
            self.pub.publish(delete_all_marker_array)
            self.pub.publish(marker_array)
            print("edges published")

    def is_within_bounding_box(self, point, box, padding=1.0):
        """Boolean return if the given point is within the given boudning box
        with an optional padding."""
        dim_x = max(box.dimensions.x, 3.0)
        dim_y = max(box.dimensions.y, 3.0)

        return (box.pose.position.x - dim_x / 2 - padding <= point.x <=
                box.pose.position.x + dim_x / 2 + padding
                and box.pose.position.y - dim_y / 2 - padding <= point.y <=
                box.pose.position.y + dim_y / 2 + padding)

    def create_central_node_and_connections(self):
        """This function creates a central node centered above the others, and
        connects all other nodes to this one."""
        central_node_marker = Marker()
        central_node_marker.header.frame_id = "world"
        central_node_marker.header.stamp = rospy.Time.now()
        central_node_marker.ns = "central_node"
        central_node_marker.id = 0
        central_node_marker.type = Marker.SPHERE
        central_node_marker.action = Marker.ADD
        central_node_marker.frame_locked = True
        central_node_marker.scale.x = self.central_node_size
        central_node_marker.scale.y = self.central_node_size
        central_node_marker.scale.z = self.central_node_size
        central_node_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        central_node_marker.lifetime = rospy.Duration(0)

        # Compute the center of all intersection_nodes
        accumulated_x, accumulated_y, accumulated_z = 0, 0, 0
        node_count = max(1, len(self.intersection_nodes))
        for node in self.intersection_nodes:
            accumulated_x += node.pose.position.x
            accumulated_y += node.pose.position.y
            accumulated_z += node.pose.position.z

        central_node_marker.pose.position.x = accumulated_x / node_count
        central_node_marker.pose.position.y = accumulated_y / node_count
        # X meters higher
        central_node_marker.pose.position.z = (accumulated_z / node_count +
                                               self.central_node_height_offset)

        # Create lines connecting the central node to all intersection_nodes
        connection_markers = []
        for node in self.intersection_nodes:
            connection_marker = Marker()
            connection_marker.header.frame_id = "world"
            connection_marker.header.stamp = rospy.Time.now()
            connection_marker.ns = "central_node_connections"
            connection_marker.id = node.id
            connection_marker.type = Marker.LINE_STRIP
            connection_marker.action = Marker.ADD
            connection_marker.frame_locked = True
            connection_marker.points = [
                central_node_marker.pose.position,
                node.pose.position,
            ]
            connection_marker.scale.x = 0.25
            connection_marker.color = ColorRGBA(0.0, 0.0, 0.0, 1.0)
            connection_marker.lifetime = rospy.Duration(0)

            connection_markers.append(connection_marker)

        return central_node_marker, connection_markers

    def create_line_marker(self, keyframe, box, scale=0.15, color=None):
        """This is a helper function to create line markers."""
        node = self.get_intersection_node(box.label)
        if node is not None:
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "keyframe_edges"
            marker.id = random.randint(0, 2**31 - 1)
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.frame_locked = True

            marker.points = [keyframe, node.pose.position]
            marker.scale.x = scale
            if color is None:
                color = node.color
            marker.color = color
            marker.lifetime = rospy.Duration(0)

            return marker
        print(f"No node found for id: {box.label}")
        return None

    def get_intersection_node(self, node_id):
        """This identifies a node with a given ID."""
        for node in self.intersection_nodes:
            if node.id == node_id:
                return node
        return None


def main() -> None:
    """Init the node, and start it."""
    rospy.init_node("intersection_keyframe_node")
    node = IntersectionKeyframeNode()

    print("Init done.")

    # pylint: disable=duplicate-code
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()


if __name__ == "__main__":
    main()
