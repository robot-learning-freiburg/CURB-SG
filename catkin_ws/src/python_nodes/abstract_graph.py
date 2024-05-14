"""This file contains functions that are needed to generate the abstract nodes
from a given lane graph."""
# pylint: disable=duplicate-code, import-error, too-many-nested-blocks, too-many-locals, too-many-statements, too-many-branches

import copy
import math
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Set

import networkx as nx
import numpy as np
import numpy.typing as npt
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN


class BoundingBoxCollection(NamedTuple):
    """This Class is used so the main file does not need to import ros
    libraries."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


def find_intersection_by_id(intersections: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
    """Returns and intersection if there is one that corresponds to the given
    index."""
    for intersection in intersections:
        if intersection["label"] == index:
            return intersection
    return {}


def is_straight(angle, threshold=10):
    """Boolean return if the angle is below a given threshold."""
    return abs(180 - angle) <= threshold


def area_triangle(point_x1, point_y1, point_x2, point_y2, point_x3, point_y3):  # pylint: disable=too-many-arguments
    """Calculates the area of a given triangle."""
    return abs((point_x1 * (point_y2 - point_y3) + point_x2 * (point_y3 - point_y1) + point_x3 *
                (point_y1 - point_y2)) / 2.0)


def remove_single_ends(pred_lanegraph):
    """This function removes single deadends from a given lane graph.

    A single dead end in this case is a branch-off of the the main
    route, that has only the length of 1
    """
    nodes_to_remove = []
    for node in pred_lanegraph.nodes():
        if pred_lanegraph.degree(node) > 2:
            neighbors = list(pred_lanegraph.successors(node)) + list(
                pred_lanegraph.predecessors(node))
            for neighbor in neighbors:
                if pred_lanegraph.degree(neighbor) == 1:
                    nodes_to_remove.append(neighbor)

    for node in nodes_to_remove:
        if node in pred_lanegraph:
            pred_lanegraph.remove_node(node)

    return pred_lanegraph


def remove_redundancy(pred_lanegraph):
    """This function removes redundant paths in the given lane graph.

    Redundancy is given if:
    A - B - C and A - C, then we can remove node B.
    It is important to always ensure that the resulting graph is still connected.
    """
    nodes_to_remove = []

    for edge1 in pred_lanegraph.edges():
        for edge2 in pred_lanegraph.edges():
            if edge1 != edge2:
                node_first = edge1[0]
                node_middle = edge1[1]
                node_last = edge2[1]

                if (pred_lanegraph.has_edge(node_first, node_middle)
                        and pred_lanegraph.has_edge(node_first, node_last)
                        and pred_lanegraph.has_edge(node_middle, node_last)):
                    if pred_lanegraph.degree(node_middle) == 2:
                        nodes_to_remove.append(node_middle)

    constellations = []
    starting_points = []

    for node_a in pred_lanegraph.nodes():
        for node_b in pred_lanegraph.successors(node_a):
            for node_c in pred_lanegraph.successors(node_a):
                if node_b != node_c:
                    for node_d in pred_lanegraph.successors(node_b):
                        if node_d in pred_lanegraph.successors(node_c):
                            constellations.append((node_a, node_b, node_c, node_d))

                            if (pred_lanegraph.degree(node_b) == 2
                                    and node_a not in starting_points):
                                nodes_to_remove.append(node_b)
                                starting_points.append(node_a)
                            elif (pred_lanegraph.degree(node_c) == 2
                                  and node_a not in starting_points):
                                nodes_to_remove.append(node_c)
                                starting_points.append(node_a)

    for node in nodes_to_remove:
        if node in pred_lanegraph:
            pred_lanegraph.remove_node(node)

    return_graph = remove_single_ends(pred_lanegraph)
    return return_graph


def generate_abstract_graph(lane_graph_original: nx.DiGraph,
                            eps: float = 18.0,
                            min_samples: float = 4):
    """This function creates the list of intersections and streets from a given
    lane graph."""
    lane_graph = copy.deepcopy(lane_graph_original)

    nodes_to_remove = []

    for edge1 in lane_graph.edges():
        for edge2 in lane_graph.edges():
            if edge1 != edge2:
                point_a = edge1[0]
                point_b = edge1[1]
                point_c = edge2[1]

                if (lane_graph.has_edge(point_a, point_b) and lane_graph.has_edge(point_a, point_c)
                        and lane_graph.has_edge(point_b, point_c)):
                    if lane_graph.degree(point_b) == 2:
                        nodes_to_remove.append(point_b)

    for node in nodes_to_remove:
        if node in lane_graph:
            lane_graph.remove_node(node)

    out_graph = copy.deepcopy(lane_graph)

    points = []

    for node in lane_graph.nodes():
        degree = lane_graph.degree(node)
        degree -= 1

        position = lane_graph.nodes[node]["pos"]

        if degree == 2:
            points.append(position)

        for index in range(degree):
            for _ in range(index):
                points.append(position)

    intersections = []

    if points:
        points_array = np.array(points)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)

        cluster_positions = defaultdict(list)

        for point, label in zip(points_array, clustering.labels_):
            cluster_positions[label].append(point)

        for label, points in cluster_positions.items():
            if label == -1:  # Skip noise points
                continue

            mean_position = np.mean(points, axis=0)

            empty_bb = BoundingBoxCollection(
                x_min=float("inf"),
                y_min=float("inf"),
                x_max=-float("inf"),
                y_max=-float("inf"),
            )
            intersections.append({
                "label": label,
                "pos": tuple(mean_position),
                "bounding_box": empty_bb
            })

        nodes_to_remove = []

        for intersection in intersections:
            intersection_position = intersection["pos"]

            find_nodes_to_remove(lane_graph, intersection, intersection_position, nodes_to_remove)
        for node in nodes_to_remove:

            if node in out_graph:
                out_graph.nodes[node]["spatial"] = "1"
            else:
                print("Node not found")

            lane_graph.remove_node(node)

    # now the intersections that have higher node degrees are filtered out
    # next step is to find remaining intersections, that appear when
    # edges from trajectories cross

    edge_intersections: List[Any] = []

    for edge1 in lane_graph.edges():
        for edge2 in lane_graph.edges():
            if edge1 == edge2:
                continue

            edge1_start_pos = lane_graph.nodes[edge1[0]]["pos"]
            edge1_end_pos = lane_graph.nodes[edge1[1]]["pos"]
            edge2_start_pos = lane_graph.nodes[edge2[0]]["pos"]
            edge2_end_pos = lane_graph.nodes[edge2[1]]["pos"]

            intersection = find_intersection((edge1_start_pos, edge1_end_pos),
                                             (edge2_start_pos, edge2_end_pos))

            angle = calculate_angle(
                np.asarray(edge1_end_pos) - np.asarray(edge1_start_pos),
                np.asarray(edge2_end_pos) - np.asarray(edge2_start_pos),
            )

            if intersection is not None:
                edge_intersections.append((intersection, angle))

    found_street_intersections: Set[Any] = set()
    spatial_tolerance = 0.1
    angle_threshold = 15

    for node in lane_graph.nodes():
        node_position = lane_graph.nodes[node]["pos"]
        for intersection, angle in edge_intersections:
            distance = euclidean_distance(node_position, intersection)
            if distance <= spatial_tolerance and abs(angle) > angle_threshold:
                found_street_intersections.add(node)

    # check if there are already intersections found around this street crossing
    # if not, add it to the clusters
    for index, node in enumerate(found_street_intersections):

        add_node = True

        node_position = lane_graph.nodes[node]["pos"]

        for cluster in intersections:
            cluster_position = cluster["pos"]
            distance = euclidean_distance(node_position, cluster_position)
            if distance <= 15:
                add_node = False
                continue
        if add_node:
            empty_bb = BoundingBoxCollection(
                x_min=float("inf"),
                y_min=float("inf"),
                x_max=-float("inf"),
                y_max=-float("inf"),
            )

            intersections.append({
                "label": index + 100,
                "pos": node_position,
                "bounding_box": empty_bb
            })

    if len(found_street_intersections) > 0:
        nodes_to_remove = []

        for index, intersection_node in enumerate(found_street_intersections):
            intersection_node_position = lane_graph.nodes[intersection_node]["pos"]

            intersection = find_intersection_by_id(intersections, index + 100)

            if intersection is not None:
                find_nodes_to_remove(
                    lane_graph,
                    intersection,
                    intersection_node_position,
                    nodes_to_remove,
                )

        for node in nodes_to_remove:
            if node in out_graph:
                out_graph.nodes[node]["spatial"] = "1"
            else:
                print("Node not found")

            lane_graph.remove_node(node)

    subgraphs = [{
        "index": index,
        "graph": lane_graph.subgraph(component)
    } for index, component in enumerate(get_disconnected_subgraphs(lane_graph))]

    streets: List[Any] = []
    connected_graphs: List[Any] = []

    for main_sub in subgraphs:
        main_graph_index = main_sub["index"]

        merge_points = []
        for new_node in main_sub["graph"].nodes():
            merge_points.append(lane_graph.nodes[new_node]["pos"])

        merged_graphs = []
        for main_node in main_sub["graph"].nodes():
            node_position = lane_graph.nodes[main_node]["pos"]
            for compare_sub in subgraphs:
                merge = False
                graph_index = compare_sub["index"]

                if graph_index in merged_graphs:
                    continue

                for compare_node in compare_sub["graph"].nodes():
                    if merge:
                        continue
                    compare_node_position = lane_graph.nodes[compare_node]["pos"]
                    distance = euclidean_distance(node_position, compare_node_position)

                    if distance <= 5:
                        # merge those graphs
                        merge = True

                        merged_graphs.append(graph_index)
                        connected_graphs.append((main_graph_index, graph_index))

                        for new_node in compare_sub["graph"].nodes():
                            merge_points.append(lane_graph.nodes[new_node]["pos"])

        add_street = True
        for conn_node_1, conn_node_2 in connected_graphs:
            if conn_node_1 == conn_node_2:
                continue
            for merged_node in merged_graphs:
                if conn_node_2 == main_graph_index and conn_node_1 == merged_node:
                    add_street = False

        if add_street:
            points_array = np.array(merge_points)
            padding = 0.1
            x_min = np.min(points_array[:, 0]) - padding
            y_min = np.min(points_array[:, 1]) - padding
            x_max = np.max(points_array[:, 0]) + padding
            y_max = np.max(points_array[:, 1]) + padding

            streets.append({
                "pos":
                np.mean(merge_points, axis=0),
                "label":
                len(streets) + 1000,
                "bounding_box":
                BoundingBoxCollection(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            })

    return intersections, streets, out_graph


def find_nodes_to_remove(lane_graph, intersection, intersection_node_position, nodes_to_remove):
    """This function identifies the nodes, that can be removed from the current
    working lane graph because a given intersection is identified."""

    for node in lane_graph.nodes():

        node_position = lane_graph.nodes[node]["pos"]
        distance = euclidean_distance(node_position, intersection_node_position)
        if distance <= 15:
            bounding_box = intersection["bounding_box"]

            padding = 0.1

            intersection["bounding_box"] = BoundingBoxCollection(
                x_min=min(node_position[0], bounding_box.x_min) - padding,
                y_min=min(node_position[1], bounding_box.y_min) - padding,
                x_max=max(node_position[0], bounding_box.x_max) + padding,
                y_max=max(node_position[1], bounding_box.y_max) + padding,
            )

            if node not in nodes_to_remove:
                nodes_to_remove.append(node)


def get_disconnected_subgraphs(lane_graph):
    """This function returns disconnected subgraphs from a given lane graph.

    This is important to identify streets.
    """
    # Find weakly connected components
    weakly_connected_components = list(nx.weakly_connected_components(lane_graph))

    # Create a subgraph for each weakly connected component
    subgraphs = [lane_graph.subgraph(nodes).copy() for nodes in weakly_connected_components]

    return subgraphs


def find_intersection(edge1, edge2):
    """This function retruns if there is a geometrical intersection of two
    given edges/lines."""
    line1 = LineString([edge1[0], edge1[1]])
    line2 = LineString([edge2[0], edge2[1]])
    intersection = line1.intersection(line2)
    if intersection.is_empty:
        return None

    intersection_point = np.array(intersection.coords[0])

    if (np.allclose(intersection_point, edge1[0]) or np.allclose(intersection_point, edge1[1])
            or np.allclose(intersection_point, edge2[0])
            or np.allclose(intersection_point, edge2[1])):
        return None

    return intersection_point


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


def laplacian_smoothing_over_degrees(lane_graph, iterations=1):
    """This function returns a laplacian smoothed graph."""
    for _ in range(iterations):
        for node in lane_graph.nodes:
            degree_sum = sum(lane_graph.degree(neighbor) for neighbor in lane_graph.neighbors(node))
            lane_graph.nodes[node]["smoothed_degree"] = (degree_sum / lane_graph.degree(node)
                                                         if lane_graph.degree(node) > 0 else 0)
        for node in lane_graph.nodes:
            lane_graph.nodes[node]["degree"] = lane_graph.nodes[node]["smoothed_degree"]
    return lane_graph


def euclidean_distance(point1, point2):
    """This function returns the euclidean distance of 2 points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def create_street_nodes(lane_graph, clusters):
    """This function returns a list of street nodes, when the list of clusters
    is given."""
    abstract_street_nodes = []

    # Iterate through all pairs of clusters
    for cluster_index, cluster1 in enumerate(clusters):
        for cluster_pair_index in range(cluster_index + 1, len(clusters)):

            cluster2 = clusters[cluster_pair_index]

            # Find the shortest path between any pair of nodes in the two clusters
            shortest_path = None
            min_distance = float("inf")
            for node1 in cluster1:
                for node2 in cluster2:
                    try:
                        path = nx.shortest_path(lane_graph, node1, node2)

                        distance = (len(path) - 1
                                    )  # The distance is the number of edges in the path
                        if distance < min_distance:
                            min_distance = distance
                            shortest_path = path
                    except nx.NetworkXNoPath:
                        pass

            # Create a new abstract street node for each edge in the shortest path
            if shortest_path is not None:
                for index in range(len(shortest_path) - 1):
                    start_node = lane_graph.nodes[shortest_path[index]]["pos"]
                    end_node = lane_graph.nodes[shortest_path[index + 1]]["pos"]
                    abstract_street_node = {
                        "pos": (
                            (start_node[0] + end_node[0]) / 2,
                            (start_node[1] + end_node[1]) / 2,
                        )
                    }
                    abstract_street_nodes.append(abstract_street_node)

    return abstract_street_nodes


def identify_intersection_nodes(lane_graph):
    """This function returns a list of nodes where the degree property is >
    1.5."""
    return [node for node, data in lane_graph.nodes(data=True) if data["degree"] > 1.5]


def cluster_intersections(lane_graph, intersection_nodes):
    """This function performs a clustering of the given intersection nodes, so
    we ensure that each intersection is only counted once."""
    # Create a subgraph with only the intersection nodes
    subgraph = lane_graph.subgraph(intersection_nodes)

    # Find connected components (clusters) in the subgraph
    clusters = list(nx.connected_components(subgraph.to_undirected()))

    # Calculate the center of each cluster and store it as a new abstract intersection node
    abstract_intersection_nodes = []
    for cluster in clusters:
        avg_x = sum(lane_graph.nodes[node]["pos"][0] for node in cluster) / len(cluster)
        avg_y = sum(lane_graph.nodes[node]["pos"][1] for node in cluster) / len(cluster)
        abstract_intersection_nodes.append({"pos": (avg_x, avg_y), "buffer": 10})
    return abstract_intersection_nodes


def add_node_degree_to_graph(lane_graph: nx.DiGraph):
    """This function adds the degree property to the lane graph nodes."""
    # Add the degree of each node as a property
    for node in lane_graph.nodes:
        lane_graph.nodes[node]["degree"] = lane_graph.degree(node)

    return lane_graph
