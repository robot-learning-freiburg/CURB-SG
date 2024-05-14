"""This module is a node, that reads the traffic sign data from the semantic
map and generates bounding boxes around the traffic signs by using a DBScan
clustering algorithm."""
# pylint: disable=duplicate-code, import-error, too-many-arguments, too-many-locals, too-many-lines, no-name-in-module, too-many-instance-attributes, too-many-branches, too-many-statements
import copy
import hashlib
import math
import time
from random import randint, seed
from typing import Any, Dict, List, NamedTuple, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
import rospy
from geometry_msgs.msg import Point

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from visualization_msgs.msg import Marker, MarkerArray
from hdl_graph_slam.msg import DynObservationArray_msg
import abstract_graph as ab
import aggregate as ag


class PointXYZ(NamedTuple):
    """Dataclass for the xyz point."""

    x: float
    y: float
    z: float


class Observation(NamedTuple):
    """Dataclass for a single observation."""

    stamp: float
    point: PointXYZ
    vehicle_id: int

    def __str__(self) -> str:
        # pylint: disable=line-too-long
        return f"vID: {self.vehicle_id}, P: ({self.point.x}, {self.point.y},  {self.point.z}), t: {self.stamp}"


def get_unique_color(vehicle_id: int) -> Tuple[float, float, float]:
    """Creates a consistent unique color for a given vehicle id #"""
    # Convert the ID into a unique hexadecimal value using a hashing function
    md5_hash = hashlib.md5(str(vehicle_id).encode("utf-8"))
    hex_value = md5_hash.hexdigest()[:6]  # Use the first 6 characters of the hex value

    # Convert the hexadecimal value into RGB values
    red, green, blue = (
        int(hex_value[:2], 16) / 255.0,
        int(hex_value[2:4], 16) / 255.0,
        int(hex_value[4:], 16) / 255.0,
    )
    return red, green, blue


def create_point(pos: Tuple[float, float], height_offset: float = 0.0) -> Point:
    """Outsourced method to return a ros point from a numpy array with an
    inserted height offset."""
    point = Point()
    point.x = pos[0]
    point.y = pos[1]
    point.z = 2.0 + height_offset
    return point


def calculate_angle(vector_1: npt.NDArray[np.single], vector_2: npt.NDArray[np.single]) -> float:
    """Function to calculate the angle between 2 vectors."""
    # Calculate the dot product of the vectors
    dot_product = np.dot(vector_1, vector_2)

    # Calculate the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(vector_1)
    magnitude_v2 = np.linalg.norm(vector_2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    # Calculate the angle between the vectors
    angle = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))

    # Convert the angle from radians to degrees
    return float(np.degrees(angle))


def angle_between_edges(graph, edge_u, edge_v):
    """Returns the angle between 2 graph edges."""
    if len(list(graph.predecessors(edge_u))) > 0:
        edge_p = list(graph.predecessors(edge_u))[0]
    else:
        return 0.0

    if len(list(graph.successors(edge_v))) > 0:
        edge_s = list(graph.successors(edge_v))[0]
    else:
        return 0.0
    u_pos = graph.nodes[edge_u]["pos"]
    v_pos = graph.nodes[edge_v]["pos"]
    p_pos = graph.nodes[edge_p]["pos"]
    s_pos = graph.nodes[edge_s]["pos"]
    prev_vec = (u_pos[0] - p_pos[0], u_pos[1] - p_pos[1])
    next_vec = (s_pos[0] - v_pos[0], s_pos[1] - v_pos[1])
    dot_product = prev_vec[0] * next_vec[0] + prev_vec[1] * next_vec[1]
    prev_norm = math.sqrt(prev_vec[0]**2 + prev_vec[1]**2)
    next_norm = math.sqrt(next_vec[0]**2 + next_vec[1]**2)
    cos_angle = dot_product / (prev_norm * next_norm)
    angle = math.degrees(math.acos(cos_angle))
    return angle


def delete_marker_array(pub, old_marker_array):
    """This function publishes a transparent marker array to delete the
    published ones."""
    # Create a MarkerArray object
    marker_array = MarkerArray()

    # Assume old_marker_array contains markers you've previously added
    for marker in old_marker_array.markers:
        marker.color.a = 0.0
        marker_array.markers.append(marker)

    # Publish MarkerArray to clear markers
    pub.publish(marker_array)


def filter_graph(graph):
    """This function filters the graph for nodes that are closer than 0.5m."""
    # Filter out nodes that are closer than 0.5m
    for edge_u, edge_v in list(graph.edges):
        u_pos = graph.nodes[edge_u]["pos"]
        v_pos = graph.nodes[edge_v]["pos"]
        dist = math.sqrt((u_pos[0] - v_pos[0])**2 + (u_pos[1] - v_pos[1])**2)
        if dist < 0.5:
            graph.remove_edge(edge_u, edge_v)

    # Filter out any isolated nodes
    graph.remove_nodes_from(list(nx.isolates(graph)))

    return graph


def filter_observations(
    observations: List[Observation],
    angle_threshold: float = 45.0,
    distance_threshold: float = 3.5,
    distance_new_graph: float = 12.0,
    minimum_node_distance: float = 0.4,
    minimum_number_of_points: int = 3,
    minimum_number_of_points_subgraph: int = 4,
    maximum_number_of_skips: int = 0,
    time_difference_for_new_path: float = 15.0,
) -> List[List[Observation]]:
    """This function filters the points of one vehicle to reject outliers."""
    # this is the path of an agent, so the list is already filtered
    if observations[0].vehicle_id < 10:
        return [observations]

    pre_filtered_observations = dbscan_filter(observations, eps=3.5, min_samples=2)

    positions = np.array([(obj.point.x, obj.point.y, obj.point.z)
                          for obj in pre_filtered_observations])

    if positions.shape[0] == 0:
        print("The array is empty.")
        return []

    # Filter out points that are too close to each other
    clustering = DBSCAN(eps=0.9, min_samples=6).fit(positions)
    filtered_observations = []

    for obj, label in zip(pre_filtered_observations, clustering.labels_):
        if label == -1:  # Object is not part of any cluster
            filtered_observations.append(obj)

    coord_list = [(observation.point.x, observation.point.y, observation.point.z)
                  for observation in filtered_observations]
    stamp_list = [observation.stamp for observation in filtered_observations]
    coord_array: npt.NDArray[np.single] = np.array(coord_list)

    if len(coord_array) < minimum_number_of_points:
        return []

    accumulated_distance: float = 0.0
    accumulated_distance_new: float = 0.0
    anchor_vector: npt.NDArray[np.single] = coord_array[1] - coord_array[0]
    anchor_point: npt.NDArray[np.single] = coord_array[1]
    anchor_point_stamp: float = stamp_list[0]

    skipped_points: int = 0

    filtered_observation_list: List[List[Observation]] = [[]]

    for index, coordinate in enumerate(coord_array):

        if index < 2:
            continue

        vector = coordinate - anchor_point
        vector_magnitude = float(np.linalg.norm(vector))
        time_diff = stamp_list[index] - anchor_point_stamp
        length_time_metric = vector_magnitude * time_diff

        if length_time_metric < 0.1:
            # this is an observation of the same place just a timestep later
            continue

        angle = calculate_angle(anchor_vector, vector)

        if skipped_points > maximum_number_of_skips:
            anchor_vector = coord_array[index - 1] - coord_array[index - 2]
            angle = calculate_angle(anchor_vector, vector)

        if abs(angle) < angle_threshold:
            skipped_points = 0

            if vector_magnitude < distance_threshold:
                skipped_points += 1
                continue

            if time_diff > time_difference_for_new_path:
                anchor_point_stamp = stamp_list[index]

                if len(filtered_observation_list[-1]) < minimum_number_of_points_subgraph:
                    filtered_observation_list[-1] = []
                else:
                    filtered_observation_list.append([])
                continue

            anchor_vector = coord_array[index] - coord_array[index - 1]
            anchor_point = coord_array[index]
            anchor_point_stamp = stamp_list[index]

            if vector_magnitude > distance_new_graph:

                if len(filtered_observation_list[-1]) < minimum_number_of_points_subgraph:
                    filtered_observation_list[-1] = []
                else:
                    filtered_observation_list.append([])
                filtered_observation_list[-1].append(filtered_observations[index])
                continue

            accumulated_distance_new += vector_magnitude
            if accumulated_distance_new > accumulated_distance + minimum_node_distance:
                accumulated_distance = accumulated_distance_new
                filtered_observation_list[-1].append(filtered_observations[index])
        else:
            skipped_points += 1

    if len(coord_array) == 1:
        return []

    return filtered_observation_list


def dbscan_filter(observations, eps, min_samples):
    """This is a prefilter, that already runs a DBSCAN clustering to remove
    some obviously clustered objects."""
    pre_positions = np.array([(obj.point.x, obj.point.y, obj.point.z) for obj in observations])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pre_positions)
    pre_filtered_observations = []
    for obj, label in zip(observations, clustering.labels_):
        if label == -1:  # Object is not part of any cluster
            pre_filtered_observations.append(obj)
        else:
            cluster_size = np.sum(clustering.labels_ == label)
            if cluster_size > 2:  # Cluster has more than 2 objects
                pre_filtered_observations.append(obj)

    return pre_filtered_observations


def delete_all_displayed_markerarrays(pub):
    """Another function to clear the displayed markers."""
    marker = Marker()
    marker.action = Marker.DELETEALL
    delete_all_marker_array = MarkerArray()
    delete_all_marker_array.markers.append(marker)
    pub.publish(delete_all_marker_array)


def create_bbox(header, box, label, value, height_offset):
    """This is a generic function to create bounding box messages."""

    bbox = BoundingBox()
    bbox.header = header
    bbox.value = value
    bbox.label = label
    pos_x = (box.x_min + box.x_max) / 2.0
    pos_y = (box.y_min + box.y_max) / 2.0

    bbox.pose.position.x = pos_x
    bbox.pose.position.y = pos_y
    bbox.pose.position.z = height_offset
    bbox.pose.orientation.w = 1.0

    dim_x = max(7.5, abs(box.x_max - box.x_min))
    dim_y = max(7.5, abs(box.y_max - box.y_min))

    bbox.dimensions.x = dim_x
    bbox.dimensions.y = dim_y
    bbox.dimensions.z = 0.1
    return bbox


def publish_intersection_nodes(graph, publisher, height=35.0):
    """This function publishes a graph on a given publisher."""
    marker_array = MarkerArray()

    marker_id = 0

    for _, attributes in graph.nodes(data=True):
        if "spatial" in attributes.keys():
            marker = Marker()
            marker.header.frame_id = "world"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.id = marker_id
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 0.8
            marker.color.g = 0.0
            marker.color.b = 0.0

            pos = attributes.get("pos", [0, 0])
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = height

            marker_array.markers.append(marker)
            marker_id += 1

    publisher.publish(marker_array)


def create_edge_marker(
    edge_id: int,
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    last_stamp: float,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    height_offset: float = 0.0,
    alpha: float = 1.0,
    size: float = 0.1,
    make_flat: bool = False,
    start_id: int = None,
    end_id: int = None,
) -> Marker:
    """Returns an edge marker that can be added to the MarkerArray."""
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = last_stamp
    seed(edge_id)
    marker.id = edge_id + randint(0, 1000000)
    marker.type = Marker.ARROW if not make_flat else Marker.LINE_STRIP
    marker.points.append(create_point(point1, height_offset))
    marker.points.append(create_point(point2, height_offset))
    marker.pose.orientation.w = 1.0
    marker.scale.x = size
    marker.scale.y = size * 2
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = alpha
    marker.ns = "edges"

    if start_id is not None:
        marker.text = f"{start_id};{end_id}"
    # marker.lifetime = rospy.Duration.from_sec(15)
    return marker


def create_node_marker(
    obs_id: int,
    pos: Tuple[float, float],
    last_stamp: float,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    size: float = 1.0,
    alpha: float = 1.0,
    height_offset: float = 0.0,
    make_flat: bool = False,
) -> Marker:
    """Returns a point marker, that can be inserted into a MarkerArray."""
    node_marker = Marker()
    node_marker.header.frame_id = "world"
    node_marker.header.stamp = last_stamp
    node_marker.type = Marker.SPHERE
    node_marker.ns = "nodes"
    seed(1)
    node_marker.id = obs_id
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
    node_marker.scale.z = size if not make_flat else 0.1
    node_marker.color.r = color[0]
    node_marker.color.g = color[1]
    node_marker.color.b = color[2]
    node_marker.color.a = alpha
    # node_marker.lifetime = rospy.Duration.from_sec(15)
    return node_marker


def visualize_bounding_boxes(
    intersection_bounding_boxes: List[ab.BoundingBoxCollection],
    street_bounding_boxes: List[ab.BoundingBoxCollection],
    last_stamp,
    sequential_id,
    height_offset,
    pub,
    pub_intersections,
    pub_streets,
):
    """Publishes a BoundingBoxArray message to visualize a list of bounding
    boxes in RViz.

    Args:
        bounding_boxes: A list of BoundingBox objects.
    """

    # Create the BoundingBoxArray message
    bbox_array = BoundingBoxArray()
    bbox_array.header.frame_id = "world"  # Change to the appropriate frame
    bbox_array.header.stamp = last_stamp
    bbox_array.header.seq = sequential_id

    bbox_array_intersections = BoundingBoxArray()
    bbox_array_intersections.header.frame_id = (
        "world"  # Change to the appropriate frame
    )
    bbox_array_intersections.header.stamp = last_stamp
    bbox_array_intersections.header.seq = sequential_id

    bbox_array_streets = BoundingBoxArray()
    bbox_array_streets.header.frame_id = "world"  # Change to the appropriate frame
    bbox_array_streets.header.stamp = last_stamp
    bbox_array_streets.header.seq = sequential_id

    # Iterate through the bounding boxes and create a BoundingBox message for each one
    print("Intersections:")
    for box_obj in intersection_bounding_boxes:
        # Add the bounding box to the array
        if np.isinf((box_obj["bounding_box"].x_min)):
            continue

        bbox_array.boxes.append(
            create_bbox(
                header=bbox_array.header,
                box=box_obj["bounding_box"],
                label=box_obj["label"],
                value=0,
                height_offset=height_offset,
            ))
        bbox_array_intersections.boxes.append(
            create_bbox(
                header=bbox_array_intersections.header,
                box=box_obj["bounding_box"],
                label=box_obj["label"],
                value=0,
                height_offset=height_offset,
            ))

    print("Streets:")
    for box_obj in street_bounding_boxes:

        # Add the bounding box to the array
        bbox_array.boxes.append(
            create_bbox(
                header=bbox_array.header,
                box=box_obj["bounding_box"],
                label=box_obj["label"],
                value=1,
                height_offset=height_offset,
            ))
        bbox_array_streets.boxes.append(
            create_bbox(
                header=bbox_array_streets.header,
                box=box_obj["bounding_box"],
                label=box_obj["label"],
                value=1,
                height_offset=height_offset,
            ))

    # Publish the BoundingBoxArray message
    pub.publish(bbox_array)
    pub_intersections.publish(bbox_array_intersections)
    pub_streets.publish(bbox_array_streets)

    print(f"""all: {len(bbox_array.boxes)}
            int: {len(bbox_array_intersections.boxes)}
            street: {len(bbox_array_streets.boxes)}""")


def visualize_nodes(
    intersection_list: List[Any],
    street_list: List[Any],
    publisher: Any,
    last_stamp: float,
    maximum_number_of_intersection_nodes,
    maximum_number_of_street_nodes,
    height_offset: float = 25.0,
    alpha: float = 1.0,
    size: float = 3.0,
) -> Tuple[int, int]:
    """This function publishes an array of small bounding boxes that represents
    the given observations."""

    # Create MarkerArray message
    marker_array = MarkerArray()
    node_count_intersections = 0
    node_count_streets = 0

    # Publish a DELETEALL action to remove all markers
    del_marker_array = MarkerArray()
    delete_marker = Marker()
    delete_marker.action = Marker.DELETEALL
    del_marker_array.markers = [delete_marker]

    # Create Marker messages for nodes
    for obs in street_list:
        pos = obs["pos"]
        label = obs["label"]

        print(label)

        node_marker = create_node_marker(
            obs_id=label,
            pos=pos,
            last_stamp=last_stamp,
            color=(0.0, 0.5, 1.0),
            size=size / 1.5,
            height_offset=height_offset,
            alpha=alpha,
        )
        marker_array.markers.append(node_marker)
        node_count_streets += 1

    while maximum_number_of_street_nodes > node_count_streets:
        # generate transparent nodes until ids are replaced

        node_marker = create_node_marker(
            obs_id=node_count_streets + 1000,
            pos=pos,
            last_stamp=last_stamp,
            color=(0.0, 0.5, 1.0),
            size=0.0,
            height_offset=height_offset,
            alpha=0.0,
        )
        marker_array.markers.append(node_marker)
        node_count_streets += 1
        print("added transparent street node")

    # Create Marker messages for nodes
    for obs in intersection_list:
        pos = obs["pos"]
        label = obs["label"]

        node_marker = create_node_marker(
            obs_id=label,
            pos=pos,
            last_stamp=last_stamp,
            color=(1.0, 0.8, 0.0),
            size=size,
            height_offset=height_offset,
            alpha=alpha,
        )
        marker_array.markers.append(node_marker)
        node_count_intersections += 1

    while maximum_number_of_intersection_nodes > node_count_intersections:
        # generate transparent nodes until ids are replaced

        node_marker = create_node_marker(
            obs_id=node_count_intersections,
            pos=pos,
            last_stamp=last_stamp,
            color=(0.0, 0.5, 1.0),
            size=0.0,
            height_offset=height_offset,
            alpha=0.0,
        )
        marker_array.markers.append(node_marker)
        node_count_intersections += 1
        print("added transparent intersection node")

    publisher.publish(marker_array)
    print(f"observations published (int: {node_count_intersections}) (str: {node_count_streets})")
    return node_count_intersections, node_count_streets


class LaneGraphBuilder:
    """This class is the node that clusters the traffic sign point clouds and
    generates the bounding boxes for all signs."""

    def __init__(self, test_mode: bool = False) -> None:

        self.working = False
        self.last_stamp = 0.0

        # intersection street nodes height
        self.lane_graph_intersection_node_height = 90

        # lane graph height
        self.lane_graph_height_offset = 40

        if not test_mode:
            self.observation_sub = rospy.Subscriber(
                "/map_server/observation_array",
                DynObservationArray_msg,
                self.observation_callback,
            )

        self.new_lane_graph_pub = rospy.Publisher("/new_graph", MarkerArray, queue_size=1)
        self.lane_graph_pub = rospy.Publisher("/lane_graph", MarkerArray, queue_size=1)
        self.dev_point_pub = rospy.Publisher("/dev_points", MarkerArray, queue_size=2)
        self.nodes_pub = rospy.Publisher("/intersection_street_nodes", MarkerArray, queue_size=2)
        self.smoothed_lane_graph_pub = rospy.Publisher("/lane_graph_smooth",
                                                       MarkerArray,
                                                       queue_size=1)
        self.smoothed_lane_graph_pub1 = rospy.Publisher("/lane_graph_smooth1",
                                                        MarkerArray,
                                                        queue_size=1)

        self.agent_path_pub = rospy.Publisher("/agent_path", MarkerArray, queue_size=1)
        self.lane_graph_intersection_node_pub = rospy.Publisher("/lane_graph_intersection_nodes",
                                                                MarkerArray,
                                                                queue_size=1)

        self.bb_pub = rospy.Publisher("/node_bounding_boxes", BoundingBoxArray, queue_size=2)
        self.bb_pub_intersection = rospy.Publisher("/node_bounding_boxes_intersection",
                                                   BoundingBoxArray,
                                                   queue_size=2)
        self.bb_pub_street = rospy.Publisher("/node_bounding_boxes_street",
                                             BoundingBoxArray,
                                             queue_size=2)

        self.current_nodes = MarkerArray()

        self.maximum_number_of_intersection_nodes = 0
        self.maximum_number_of_street_nodes = 0

        self.sequential_id = 0

    def create_digraph(self, vehicles: Dict[int, List[List[Observation]]]) -> nx.DiGraph:
        """This function creates a networkx digraph object from the given
        vehicle trajectory points."""

        lane_graph = nx.DiGraph()
        lane_graph.clear()
        new_graphs = []
        counter = 0
        for _, observations_list in tqdm(vehicles.items()):
            counter += 1

            if len(observations_list) > 0:
                if isinstance(observations_list[0], Observation):
                    observations_list = [observations_list]

            for obs_index, observations in enumerate(observations_list):
                if len(observations) < 3:
                    continue

                new_graphs.append(nx.DiGraph())

                # Add nodes to the graph
                for index, observation in enumerate(observations):

                    new_graphs[-1].add_node(
                        observation.vehicle_id * 100000 + obs_index * 1000 + index,
                        pos=np.array([observation.point.x, observation.point.y]),
                    )

                    if index + 1 != len(observations):
                        new_graphs[-1].add_edge(
                            observation.vehicle_id * 100000 + obs_index * 1000 + index,
                            observation.vehicle_id * 100000 + obs_index * 1000 + index + 1,
                        )

                self.visualize_lane_graph(
                    lane_graph=new_graphs[-1],
                    publisher=self.new_lane_graph_pub,
                    color=(1.0, 0, 0),
                    height_offset=self.lane_graph_height_offset - 1.5,
                    size=0.5,
                )

                lane_graph, _ = ag.aggregate(lane_graph, new_graphs[-1])
                if not isinstance(lane_graph, nx.DiGraph):
                    print("agg: lane not instance")

        lane_graph = ag.remove_triangle_constellations(lane_graph)

        if len(lane_graph.nodes()) > 0:
            smooth_graph = ag.laplacian_smoothing(copy.deepcopy(lane_graph),
                                                  gamma=0.03,
                                                  iterations=3)
        else:
            smooth_graph = lane_graph

        delete_all_displayed_markerarrays(self.smoothed_lane_graph_pub)

        self.visualize_lane_graph(
            lane_graph=smooth_graph,
            color=(0.5, 1.0, 0.0),
            publisher=self.smoothed_lane_graph_pub,
            height_offset=self.lane_graph_height_offset,
            size=0.9,
        )

        graph_removed_doubles = ag.merge_closeby_nodes(copy.deepcopy(smooth_graph))
        print(f"edges: {len(smooth_graph.edges)}")

        return smooth_graph, graph_removed_doubles

    def process_data(self, data: DynObservationArray_msg) -> None:
        """Outsourced method."""
        self.last_stamp = data.header.stamp
        vehicles: Dict[int, List[Observation]] = {}
        for observation in data.observations:
            if (not observation.header.stamp.to_sec() > 0 or observation.header.seq == 999999):
                continue
            obs_object = Observation(
                vehicle_id=observation.header.seq,
                point=PointXYZ(
                    x=observation.pose.position.x,
                    y=observation.pose.position.y,
                    z=observation.pose.position.z,
                ),
                stamp=observation.header.stamp.to_sec(),
            )

            # pylint: disable=consider-iterating-dictionary
            if observation.header.seq * 100 not in vehicles.keys():
                vehicles[observation.header.seq * 100] = [obs_object]
            else:
                vehicles[observation.header.seq * 100].append(obs_object)

        self.publish_positions_as_markerarray(vehicles.values(),
                                              height=self.lane_graph_height_offset - 6.0)

        lane_graph_filtered_dict = {}

        # sort the arrays according to the timestamps
        for vehicle_id, observations in sorted(vehicles.items()):

            observations = sorted(observations, key=lambda lambda_obs: lambda_obs.stamp)

            filtered_observations = filter_observations(observations)

            if observations[0].vehicle_id < 10:
                agent_path = nx.DiGraph()
                agent_path.clear()
                for obs_index, agent_observation in enumerate(filtered_observations[0]):

                    agent_path.add_node(
                        agent_observation.vehicle_id * 100000 + obs_index,
                        pos=np.array([agent_observation.point.x, agent_observation.point.y]),
                    )

                    if obs_index + 1 != len(filtered_observations[0]):
                        agent_path.add_edge(
                            agent_observation.vehicle_id * 100000 + obs_index,
                            agent_observation.vehicle_id * 100000 + obs_index + 1,
                        )
                self.visualize_lane_graph(
                    lane_graph=agent_path,
                    publisher=self.agent_path_pub,
                    color=(0.0, 1.0, 0.0),
                    height_offset=self.lane_graph_height_offset - 0.2,
                    size=2.5,
                    make_flat=True,
                )
            else:
                self.publish_positions_as_markerarray(
                    filtered_observations,
                    height=self.lane_graph_height_offset - 4.0,
                    size=1.0,
                )

            lane_graph_filtered_dict[vehicle_id] = filtered_observations

        _, pred_lanegraph_removed_doubles = self.create_digraph(lane_graph_filtered_dict)

        red1 = ab.remove_redundancy(pred_lanegraph_removed_doubles)
        reduced_graph = ab.remove_single_ends(red1)

        delete_all_displayed_markerarrays(self.lane_graph_pub)

        print("Generating nodes..")

        eps = 15.0
        min_samples = 4

        intersections, streets, annotated_graph = ab.generate_abstract_graph(
            reduced_graph, eps=eps, min_samples=min_samples)

        publish_intersection_nodes(
            annotated_graph,
            self.lane_graph_intersection_node_pub,
            height=self.lane_graph_intersection_node_height,
        )

        print(f"Intersections:  {len(intersections)} ")
        print(f"Streets:        {len(streets)}  ")

        self.visualize_lane_graph(
            lane_graph=annotated_graph,
            color=(0.5, 0.0, 0.0),
            publisher=self.lane_graph_pub,
            height_offset=self.lane_graph_height_offset + 2.5,
            size=1.5,
            annotated=True,
        )

        (
            self.maximum_number_of_intersection_nodes,
            self.maximum_number_of_street_nodes,
        ) = visualize_nodes(
            intersection_list=intersections,
            street_list=streets,
            publisher=self.nodes_pub,
            last_stamp=self.last_stamp,
            maximum_number_of_intersection_nodes=self.maximum_number_of_intersection_nodes,
            maximum_number_of_street_nodes=self.maximum_number_of_street_nodes,
            size=4.0,
            height_offset=self.lane_graph_intersection_node_height,
        )

        visualize_bounding_boxes(
            intersection_bounding_boxes=intersections,
            street_bounding_boxes=streets,
            pub=self.bb_pub,
            last_stamp=self.last_stamp,
            sequential_id=self.sequential_id,
            height_offset=self.lane_graph_height_offset - 1.5,
            pub_intersections=self.bb_pub_intersection,
            pub_streets=self.bb_pub_street,
        )

    def visualize_lane_graph(
        self,
        lane_graph: nx.DiGraph,
        publisher: Any,
        color: Tuple[float, float, float] = (1, 1, 1),
        height_offset: float = 0.0,
        alpha: float = 1.0,
        size: float = 0.20,
        make_flat: bool = False,
        annotated: bool = False,
    ) -> Any:
        """This function publishes an array of small bounding boxes that
        represents the given observations."""

        # Create MarkerArray message
        marker_array = MarkerArray()

        if not isinstance(lane_graph, nx.DiGraph):
            print("not instance")
            return marker_array

        node_count = 0

        annotated_intersection_color = (1.0, 0.8, 0.0)
        annotated_steet_color = (0.0, 0.5, 1.0)

        # Create Marker messages for nodes
        if not make_flat:
            for node in lane_graph.nodes(data=True):

                node_marker = create_node_marker(
                    obs_id=node[0],
                    pos=(node[1]["pos"][0], node[1]["pos"][1]),
                    last_stamp=self.last_stamp,
                    color=color if not annotated else annotated_intersection_color
                    if node[1].get("spatial") == "1" else annotated_steet_color,
                    size=size * 1.0,
                    height_offset=height_offset,
                    alpha=alpha,
                )
                marker_array.markers.append(node_marker)
                node_count += 1

        # Create Marker messages for edges

        edge_color = color
        if not make_flat:
            edge_color = (color[0] / 2, color[1] / 2, color[2] / 2)

        # if not make_flat:
        for index, edge in enumerate(lane_graph.edges(data=True)):
            edge_color = (color[0] / 2, color[1] / 2, color[2] / 2)

            if annotated:
                edge_color = (
                    annotated_intersection_color if lane_graph.nodes[edge[0]].get("spatial") == "1"
                    or lane_graph.nodes[edge[1]].get("spatial") == "1" else annotated_steet_color)

            marker = create_edge_marker(
                edge_id=edge[0] + index,
                point1=(
                    lane_graph.nodes[edge[0]]["pos"][0],
                    lane_graph.nodes[edge[0]]["pos"][1],
                ),
                point2=(
                    lane_graph.nodes[edge[1]]["pos"][0],
                    lane_graph.nodes[edge[1]]["pos"][1],
                ),
                last_stamp=self.last_stamp,
                color=edge_color,
                height_offset=height_offset,
                size=size / 1.8,
                alpha=alpha,
                make_flat=make_flat,
                start_id=edge[0],
                end_id=edge[1],
            )
            marker_array.markers.append(marker)

        publisher.publish(marker_array)
        return marker_array

    def publish_positions_as_markerarray(self, positions, height=0.0, size=0.3):
        """Publish an array of 2D positions as a MarkerArray in ROS.

        :param positions: Array of 2D positions as (x, y) tuples or
            lists.
        """

        # Create a MarkerArray message
        marker_array = MarkerArray()

        # Loop through the positions and create a marker for each position
        for list_index, observation_list in enumerate(positions):
            for observation in observation_list:
                marker = create_node_marker(
                    obs_id=np.random.randint(1, 1000000),
                    pos=(observation.point.x, observation.point.y),
                    last_stamp=self.last_stamp,
                    color=(
                        0.0,
                        list_index / len(positions),
                        1.0 - (list_index / len(positions)),
                    ),
                    size=size,
                    height_offset=height,
                )

                # Add the marker to the MarkerArray
                marker_array.markers.append(marker)

        self.dev_point_pub.publish(marker_array)

    def observation_callback(self, data: Any) -> None:
        """This is the callback that gets the data and the index of the agent
        processes the data and then republishes again."""
        if self.working:
            print("Rejected")
            return

        self.working = True
        self.sequential_id += 1

        print()
        print(f"""New observations received: {time.strftime('%H:%M:%S')}
                new seq-id: {self.sequential_id}""")

        start_time = time.time()

        self.process_data(data)

        print(f"Calculation time: {((time.time() - start_time)):.3f}s")
        self.working = False

        self.observation_sub.unregister()
        self.observation_sub = rospy.Subscriber(
            "/map_server/observation_array",
            DynObservationArray_msg,
            self.observation_callback,
        )

    def spin(self) -> None:
        """This in not really needed but to keep the node alive."""


def main() -> None:
    """Init the node, and start it."""

    rospy.init_node("lane_graph_node")
    node = LaneGraphBuilder()

    print("Init done.")

    # pylint: disable=duplicate-code
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()


if __name__ == "__main__":
    main()
