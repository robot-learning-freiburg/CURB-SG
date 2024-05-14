"""This is a collection of functions written by Martin Buechner et al."""
# pylint: skip-file
import copy
import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Tuple

import networkx as nx
import numpy as np
import osmnx.distance
from scipy.spatial.distance import cdist
from shapely.geometry import MultiLineString


def mean_angle_abs_diff(x: float, y: float) -> float:
    period = 2 * np.pi
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]
    return float(np.abs(diff))


def laplacian_smoothing(G, gamma=0.5, iterations=3):
    """
    Smoothes the input graph for multiple iterations with a value gamma each.
    :param G: undirected input graph to smooth
    :param gamma: smoothing intensity for graph Laplacian
    :param iterations: how often the smoothing is repeated
    """
    L = nx.laplacian_matrix(nx.Graph(G)).todense()
    O = np.eye(L.shape[0]) - gamma * L

    for it in range(iterations):

        node_pos = np.array([list(G.nodes[n]['pos']) for n in G.nodes()])
        node_pos = np.dot(O, node_pos)

        # Update node positions
        for i, node in enumerate(G.nodes()):
            #if G.degree(node) == 2:
            G.nodes[node]['pos'] = np.array(node_pos[i, :]).flatten()

    return G


def find_simple_paths_within_range(G, source, target, max_len=10):
    paths = list(nx.all_simple_paths(G, source, target, max_len))
    return paths


def remove_parallel_path(G):
    parallel_paths = []
    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 != node2:
                paths = find_simple_paths_within_range(G, node1, node2)
                if len(paths) > 1:
                    print("found a parallel path")
                    parallel_paths.append(paths)

    removed_points = []
    if parallel_paths:
        path_to_remove = parallel_paths[0][1]
        for i in range(len(path_to_remove) - 1):
            G.remove_edge(path_to_remove[i], path_to_remove[i + 1])
            removed_points.append(G.nodes[path_to_remove[i]]["pos"])

    return G, removed_points


def remove_triangle_constellations(G):
    nodes_to_remove = []

    for node_0 in G.nodes():
        for node_1 in G.successors(node_0):
            successors_of_node_1 = list(G.successors(node_1))

            if len(successors_of_node_1) == 2:
                node_2, node_3 = successors_of_node_1

                # Check if there's an edge from node_3 to node_2
                if G.has_edge(node_3, node_2):
                    nodes_to_remove.append(node_3)

                    # Remove edges connected to node_3
                    G.remove_edge(node_1, node_3)
                    G.remove_edge(node_3, node_2)

    # Remove the identified node_3 nodes
    G.remove_nodes_from(nodes_to_remove)
    return G


# Aggregate function
def aggregate(
    G_agg: nx.DiGraph,
    G_new: nx.DiGraph,
    threshold_px: float = 2.0,  # 7.0
    threshold_rad: float = 0.35,
    closest_lat_thresh: float = 4.0,
    w_decay: bool = False,
) -> Tuple[nx.DiGraph, Any]:
    """
    G_agg: graph to aggregate to
    G_new: graph to aggregate
    threshold_px: threshold in pixels for merging nodes
    threshold_rad: threshold in radians for merging nodes
    closest_lat_thresh: threshold in pixels between potentially parallel edges
    w_decay: whether to decay weights of nodes in G_new (not used for now)
    """

    # Maps from agg nodes to new nodes
    merging_map: Any = defaultdict(list)

    # Add aggregation weight to new predictions
    if w_decay:
        new_in_degree = dict(G_new.in_degree(list(G_new.nodes())))
        # Check if dict is empty
        if len(new_in_degree) > 0:
            # Get key of new_in_degree dict with minimum value
            new_ego_root_node = min(new_in_degree, key=new_in_degree.get)  # type: ignore
            shortest_paths_from_root = nx.shortest_path_length(G_new, new_ego_root_node)
            for n in G_new.nodes():
                G_new.nodes[n]["weight"] = 1 - 0.05 * shortest_paths_from_root[n]
    else:
        for n in G_new.nodes():
            G_new.nodes[n]["weight"] = 1.0

    # Add edge angles to new graph
    for e in G_new.edges():
        G_new.edges[e]["angle"] = np.arctan2(
            G_new.nodes[e[1]]["pos"][1] - G_new.nodes[e[0]]["pos"][1],
            G_new.nodes[e[1]]["pos"][0] - G_new.nodes[e[0]]["pos"][0],
        )

    # Get mean of angles of edges connected to each node in G_agg
    for n in G_new.nodes():
        edge_angles_pred = [
            nx.get_edge_attributes(G_new, "angle")[(x, n)] for x in G_new.predecessors(n)
        ]
        edge_angles_succ = [
            nx.get_edge_attributes(G_new, "angle")[(n, x)] for x in G_new.successors(n)
        ]
        edge_angles = edge_angles_pred + edge_angles_succ
        edge_angles_sin = [np.sin(angle) for angle in edge_angles]
        edge_angles_cos = [np.cos(angle) for angle in edge_angles]
        mean_angle = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
        if len(edge_angles_pred) == 0 and len(edge_angles_succ) == 0:
            mean_angle = 0
        G_new.nodes[n]["mean_angle"] = mean_angle

    # What if G_agg is empty? Then just return G_new, because it's the first graph and will be used as G_agg in next iteration
    # if len(G_agg.nodes) == 0:
    if not isinstance(G_agg, nx.DiGraph):
        print("agg: gagg not instance")
        return G_new.copy(), merging_map

    # Assign angle attribute on edges of G_agg and G_new
    for e in G_agg.edges():
        G_agg.edges[e]["angle"] = np.arctan2(
            G_agg.nodes[e[1]]["pos"][1] - G_agg.nodes[e[0]]["pos"][1],
            G_agg.nodes[e[1]]["pos"][0] - G_agg.nodes[e[0]]["pos"][0],
        )

    # Get mean of angles of edges connected to each node in G_agg
    for n in G_agg.nodes():
        edge_angles_pred = [
            nx.get_edge_attributes(G_agg, "angle")[(x, n)] for x in G_agg.predecessors(n)
        ]
        edge_angles_succ = [
            nx.get_edge_attributes(G_agg, "angle")[(n, x)] for x in G_agg.successors(n)
        ]
        edge_angles = edge_angles_pred + edge_angles_succ
        edge_angles_sin = [np.sin(angle) for angle in edge_angles]
        edge_angles_cos = [np.cos(angle) for angle in edge_angles]
        mean_angle = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
        if len(edge_angles_pred) == 0 and len(edge_angles_succ) == 0:
            mean_angle = 0
        G_agg.nodes[n]["mean_angle"] = mean_angle

    # Get node name map
    node_names_agg = list(G_agg.nodes())
    node_names_new = list(G_new.nodes())

    # Get pairwise distance between nodes in G_agg and G_new
    node_pos_agg = np.array([G_agg.nodes[n]["pos"] for n in G_agg.nodes]).reshape(-1, 2)
    node_pos_new = np.array([G_new.nodes[n]["pos"] for n in G_new.nodes]).reshape(-1, 2)
    node_distances = cdist(node_pos_agg, node_pos_new, metric="euclidean")  # i: agg, j: new

    # Get pairwise angle difference between nodes in G_agg and G_new
    node_mean_ang_agg = np.array([G_agg.nodes[n]["mean_angle"] for n in G_agg.nodes]).reshape(-1, 1)
    node_mean_ang_new = np.array([G_new.nodes[n]["mean_angle"] for n in G_new.nodes]).reshape(-1, 1)
    node_mean_ang_distances = cdist(node_mean_ang_agg, node_mean_ang_new,
                                    lambda u, v: mean_angle_abs_diff(u, v))

    # Produce a pairwise thresholding that allows the construction of ROIs in terms of Euclidean distance
    # and angle difference
    position_criterium = node_distances < threshold_px
    angle_criterium = node_mean_ang_distances < threshold_rad
    criterium = position_criterium & angle_criterium

    closest_agg_nodes: Any = defaultdict()

    # Loop through all new nodes (columns indexed with j)
    for j in range(criterium.shape[1]):
        # Loop through all close agg-nodes and construct the j-specific local agg graph
        agg_j_multilines = list()

        # Get all agg-nodes that are close to new node j
        # Use orthogonal linear coordinates system to avoid problems arising from OSMnx distance calculation
        G_agg_j = nx.MultiDiGraph(crs="EPSG:3857")
        for i in range(criterium.shape[0]):
            if criterium[i, j]:  # check if agg node i is close enough to new node j
                # print(
                #     f"criterium true {j}: {node_distances[i, j]} - {node_mean_ang_distances[i, j]}")
                for e in G_agg.edges(node_names_agg[i]):
                    # Add edge to local agg graph
                    G_agg_j.add_node(
                        str(e[0]),
                        x=G_agg.nodes[e[0]]["pos"][0],
                        y=G_agg.nodes[e[0]]["pos"][1],
                    )
                    G_agg_j.add_node(
                        str(e[1]),
                        x=G_agg.nodes[e[1]]["pos"][0],
                        y=G_agg.nodes[e[1]]["pos"][1],
                    )
                    G_agg_j.add_edge(str(e[0]), str(e[1]))
                    agg_j_multilines.append((
                        (G_agg.nodes[e[0]]["pos"][0], G_agg.nodes[e[0]]["pos"][1]),
                        (G_agg.nodes[e[1]]["pos"][0], G_agg.nodes[e[1]]["pos"][1]),
                    ))
        agg_j_shapely = MultiLineString(agg_j_multilines)
        # Find the closest edge and closest_node in agg-graph to new node j
        if len(list(G_agg_j.edges)) > 0:
            closest_edge, closest_lat_dist = osmnx.distance.nearest_edges(
                G_agg_j,
                float(G_new.nodes[node_names_new[j]]["pos"][0]),
                float(G_new.nodes[node_names_new[j]]["pos"][1]),
                return_dist=True,
            )
            closest_node = osmnx.distance.nearest_nodes(
                G_agg_j,
                float(G_new.nodes[node_names_new[j]]["pos"][0]),
                float(G_new.nodes[node_names_new[j]]["pos"][1]),
                return_dist=False,
            )
            closest_node = eval(closest_node)
            closest_node_dist = np.linalg.norm(
                np.array(G_agg.nodes[closest_node]["pos"]) - G_new.nodes[node_names_new[j]]["pos"])

            if closest_lat_dist < closest_lat_thresh:
                closest_i, closest_j = eval(closest_edge[0]), eval(closest_edge[1])

                # assign second-closest to closest_node not closest_i
                if closest_i == closest_node:
                    sec_closest_node = closest_j
                else:
                    sec_closest_node = closest_i

                closest_agg_nodes[node_names_new[j]] = closest_node

                sec_closest_node_dist = np.linalg.norm(
                    np.array(G_agg.nodes[sec_closest_node]["pos"]) -
                    G_new.nodes[node_names_new[j]]["pos"])

                closest_node_dist_x = (G_agg.nodes[closest_node]["pos"][0] -
                                       G_new.nodes[node_names_new[j]]["pos"][0])
                closest_node_dist_y = (G_agg.nodes[closest_node]["pos"][1] -
                                       G_new.nodes[node_names_new[j]]["pos"][1])

                alpha = np.arccos(
                    closest_lat_dist /
                    closest_node_dist) if closest_lat_dist / closest_node_dist < 1.0 else 0.0
                beta = np.arctan(closest_node_dist_y / closest_node_dist_x)
                gamma = np.pi / 2 - alpha - beta

                sec_alpha = np.arccos(closest_lat_dist / sec_closest_node_dist)

                closest_long_dist = closest_node_dist * np.sin(alpha)
                sec_closest_long_dist = sec_closest_node_dist * np.sin(sec_alpha)

                curr_new_node = np.array(G_new.nodes[node_names_new[j]]["pos"])
                virtual_closest_lat_node = curr_new_node + closest_long_dist * np.array(
                    [-np.cos(gamma), np.sin(gamma)])
                virtual_sec_closest_lat_node = (
                    curr_new_node +
                    sec_closest_long_dist * np.array([np.cos(gamma), -np.sin(gamma)]))

                omega_closest = 1 - closest_node_dist / (closest_node_dist + sec_closest_node_dist)
                omega_sec_closest = 1 - sec_closest_node_dist / (closest_node_dist +
                                                                 sec_closest_node_dist)

                # Calculating the node weights for aggregation
                closest_node_weight = G_agg.nodes[closest_node]["weight"] + 1 if G_agg.nodes[
                    closest_node]["weight"] != -1 else -1
                closest_agg_node_weight = G_agg.nodes[closest_node]["weight"] / closest_node_weight
                closest_new_node_weight = (omega_closest * 1 / closest_node_weight)

                # Normalization of closest weights
                closest_weights_sum = closest_agg_node_weight + closest_new_node_weight
                closest_agg_node_weight = closest_agg_node_weight / closest_weights_sum
                closest_new_node_weight = closest_new_node_weight / closest_weights_sum

                second_closest_node_weight = G_agg.nodes[sec_closest_node][
                    "weight"] + 1 if G_agg.nodes[sec_closest_node]["weight"] != -1 else -1
                sec_closest_agg_node_weight = G_agg.nodes[sec_closest_node][
                    "weight"] / second_closest_node_weight
                sec_closest_new_node_weight = (omega_sec_closest * 1 / second_closest_node_weight)
                # Normalization of sec-closest weights
                sec_closest_weights_sum = (sec_closest_agg_node_weight +
                                           sec_closest_new_node_weight)
                sec_closest_agg_node_weight = (sec_closest_agg_node_weight /
                                               sec_closest_weights_sum)
                sec_closest_new_node_weight = (sec_closest_new_node_weight /
                                               sec_closest_weights_sum)

                updtd_closest_node_pos = closest_agg_node_weight * np.array(
                    G_agg.nodes[closest_node]["pos"]) + closest_new_node_weight * np.array(
                        virtual_closest_lat_node)
                updtd_sec_closest_node_pos = sec_closest_agg_node_weight * np.array(
                    G_agg.nodes[sec_closest_node]["pos"]) + sec_closest_new_node_weight * np.array(
                        virtual_sec_closest_lat_node)

                # Check if the updated node is not NaN
                if not math.isnan(updtd_closest_node_pos[0] * updtd_closest_node_pos[1]):
                    (
                        G_agg.nodes[closest_node]["pos"][0],
                        G_agg.nodes[closest_node]["pos"][1],
                    ) = (updtd_closest_node_pos[0], updtd_closest_node_pos[1])
                if not math.isnan(updtd_sec_closest_node_pos[0] * updtd_sec_closest_node_pos[1]):
                    (
                        G_agg.nodes[sec_closest_node]["pos"][0],
                        G_agg.nodes[sec_closest_node]["pos"][1],
                    ) = (updtd_sec_closest_node_pos[0], updtd_sec_closest_node_pos[1])

                # Record merging weights
                G_agg.nodes[closest_node]["weight"] += 1
                G_agg.nodes[sec_closest_node]["weight"] += 1

                merging_map[closest_node].append(node_names_new[j])
                merging_map[sec_closest_node].append(node_names_new[j])

    # What happens to all other nodes in G_new? Add them to G_agg
    mapped_new_nodes = [*merging_map.values()]
    mapped_new_nodes = [item for sublist in mapped_new_nodes for item in sublist]
    c = 0
    # print(f"G_agg o: {len(G_agg.nodes)}")
    for index, n in enumerate(G_new.nodes()):
        if n not in mapped_new_nodes:
            c += 1
            G_agg.add_node(
                n,
                pos=G_new.nodes[n]["pos"],
                weight=G_new.nodes[n]["weight"],
            )
        # else:
        # print("node is in mapped_new_nodes")
    # print(f"G_agg n: {len(G_agg.nodes)}")
    # print(f"Added nodes: {c}")
    # print(f"G_new: {len(G_new.nodes)}")

    for e in G_new.edges():
        n = e[0]
        m = e[1]

        angle = np.arctan2(
            G_new.nodes[m]["pos"][1] - G_new.nodes[n]["pos"][1],
            G_new.nodes[m]["pos"][0] - G_new.nodes[n]["pos"][0],
        )

        # Add completely new edges
        if n not in mapped_new_nodes and m not in mapped_new_nodes:
            G_agg.add_edge(n, m, angle=G_new.edges[e]["angle"])

        # Add leading edges
        if n in mapped_new_nodes and m not in mapped_new_nodes:
            angle = np.arctan2(
                G_new.nodes[m]["pos"][1] - G_agg.nodes[closest_agg_nodes[n]]["pos"][1],
                G_new.nodes[m]["pos"][0] - G_agg.nodes[closest_agg_nodes[n]]["pos"][0],
            )
            # print(closest_agg_nodes[n])
            G_agg.add_edge(closest_agg_nodes[n], m, angle=angle)

        # Add trailing edges
        if n not in mapped_new_nodes and m in mapped_new_nodes:
            angle = np.arctan2(
                G_agg.nodes[closest_agg_nodes[m]]["pos"][1] - G_new.nodes[n]["pos"][1],
                G_agg.nodes[closest_agg_nodes[m]]["pos"][0] - G_new.nodes[n]["pos"][0],
            )
            # print(closest_agg_nodes[m])
            G_agg.add_edge(n, closest_agg_nodes[m], angle=angle)
    # print(len(G_agg.nodes))
    return G_agg, merging_map


def merge_nodes(G, node1, node2):
    # compute new position as average of two node positions
    new_pos = [(p1 + p2) / 2 for p1, p2 in zip(G.nodes[node1]['pos'], G.nodes[node2]['pos'])]

    # create new node with new position
    new_node = G.number_of_nodes() + 1  # assumes nodes are integers
    G.add_node(new_node, pos=new_pos, mer=1)
    # connect new node to neighbors of old nodes
    # print()
    # print()
    # print()
    # print()
    #
    # print(f"node1: {node1}")
    # print(f"out edges: {G.out_edges(node1, data=True)}")
    # print(f"in edges: {G.in_edges(node1, data=True)}")
    # print(f"all edges: {G.edges(node1, data=True)}")
    # print(f"neighbors: {list(G.neighbors(node1))}")
    # print(f"node2: {node2}")
    # print(f"out edges: {G.out_edges(node2, data=True)}")
    # print(f"in edges: {G.in_edges(node2, data=True)}")
    # print(f"all edges: {G.edges(node2, data=True)}")
    # print(f"neighbors: {list(G.neighbors(node2))}")

    incoming_edges = []
    for edge in list(G.in_edges(node1)):
        incoming_edges.append(edge)
    for edge in list(G.in_edges(node2)):
        incoming_edges.append(edge)
    # incoming_edges.append(list(G.in_edges(node2)))
    # print(f"incoming edges: {incoming_edges}")
    incoming_nodes = [n[0] for n in incoming_edges]

    # print(outgoing_nodes)

    # print(f"newnode: {new_node}")
    # print(f"neighbors: {list(G.neighbors(new_node))}")

    outgoing_nodes = set(G.neighbors(node1)).union(G.neighbors(node2))
    # print(f"outgoing edges: {outgoing_nodes}")

    # for neighbor in set(G.neighbors(node1)).union(G.neighbors(node2)):
    #     print(f"new_node: {new_node} -> neighbor: {neighbor}")
    #     G.add_edge(new_node, neighbor)

    # also add an edge from neighbor to new_node if it existed before
    for neighbor in incoming_nodes:
        # try:
        #     this_is_a_merge_node = G.nodes[neighbor]["mer"]
        # except KeyError:
        #     this_is_a_merge_node = 0
        # if this_is_a_merge_node == 1:
        # print(f"neighbor {neighbor} is a merge node")
        # continue

        # print(f"incoming edge: neighbor: {neighbor} -> new_node: {new_node}")
        G.add_edge(neighbor, new_node, merged=2)

        # print(G.nodes[new_node])
        # print(G.nodes[new_node]["pos"])
        G[neighbor][new_node]["angle"] = np.arctan2(
            G.nodes[new_node]["pos"][1] - G.nodes[neighbor]["pos"][1],
            G.nodes[new_node]["pos"][0] - G.nodes[neighbor]["pos"][0],
        )

    for neighbor in outgoing_nodes:
        # try:
        #     this_is_a_merge_node = G.nodes[neighbor]["mer"]
        # except KeyError:
        #     this_is_a_merge_node = 0
        # if this_is_a_merge_node == 1:
        #     print(f"neighbor {neighbor} is a merge node")
        #     # continue

        # print(f"outgoing edge: new_node: {new_node} -> neighbor: {neighbor}")
        G.add_edge(new_node, neighbor, merged=1)
        G[new_node][neighbor]["angle"] = np.arctan2(
            G.nodes[neighbor]["pos"][1] - G.nodes[new_node]["pos"][1],
            G.nodes[neighbor]["pos"][0] - G.nodes[new_node]["pos"][0],
        )

    # print(f"newnode: {new_node}")
    # print(f"out edges: {G.out_edges(new_node, data=True)}")
    # print(f"in edges: {G.in_edges(new_node, data=True)}")
    # print(f"all edges: {G.edges(new_node, data=True)}")
    # print(f"neighbors: {list(G.neighbors(new_node))}")
    #
    # # remove old nodes
    # # G.remove_node(node1)
    # # G.remove_node(node2)
    # print()
    # print()
    # print()
    # print()
    return G, node1, node2

    # define a function to compute the distance between two nodes


def distance(node1, node2):
    pos1 = np.array(node1['pos'])
    pos2 = np.array(node2['pos'])
    return np.linalg.norm(pos1 - pos2)


# define a function to compute the angle of an edge
def angle(edge):
    # print(edge)
    pos1 = np.array(edge[0]['pos'])
    pos2 = np.array(edge[1]['pos'])
    return np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])


def remove_duplicate_edges(G):
    if G.is_multigraph():
        # Create a copy as a Graph or DiGraph, which automatically removes duplicate edges
        G_temp = nx.DiGraph(G) if G.is_directed() else nx.Graph(G)
        # Then convert it back to a MultiGraph or MultiDiGraph
        G_new = nx.MultiDiGraph(G_temp) if G.is_directed() else nx.MultiGraph(G_temp)
        return G_new
    else:
        # If G is not a multigraph, there's nothing to do
        return G


def normalize_angle(angle):
    normalized_angle = angle % (2 * math.pi)
    if normalized_angle > math.pi:
        normalized_angle -= 2 * math.pi
    return normalized_angle


def remove_nodes_with_zero_outdegree(G):
    # Create a copy of the nodes to iterate over, because the graph will be modified during iteration
    nodes = list(G.nodes())

    remove_nodes = []

    for node in nodes:
        # Check if this node has outdegree 0
        if G.out_degree(node) >= 2:
            # Check each neighbor of the node
            for neighbor in G.neighbors(node):
                # If the neighbor has outdegree > 2, remove the current node
                if G.out_degree(neighbor) == 0:
                    remove_nodes.append(neighbor)
                    print(
                        f"Removed node {node} because it has outdegree 0 and its neighbor {neighbor} has outdegree > 2"
                    )
                    break

    for node in remove_nodes:
        try:
            G.remove_node(node)
        except:
            pass
            # print(f"node {node} already removed")

    return G


def merge_closeby_nodes(G, distance_threshold=3.5, angle_threshold=0.4):
    print("Merging closeby nodes...")
    nodes_before_merging = len(G.nodes)
    edges_before_merging = len(G.edges)

    nodes_to_remove = []

    # iterate over all pairs of nodes
    for node1, node2 in combinations(G.nodes(data=True), 2):
        # check if nodes have spatial proximity of 0.5
        # print(f"distance: {distance(node1[1], node2[1])}")
        if node1[0] in nodes_to_remove or node2[0] in nodes_to_remove:
            # print("skipping node")
            continue

        if distance(node1[1], node2[1]) <= distance_threshold:
            # print("nodes are close")
            # check if nodes have same angle of outgoing edges

            # print(f"node1: {node1}")
            # print(f"node2: {node2}")

            skip = False

            for edge1, edge2 in zip(list(G.out_edges(node1[0], data=True)),
                                    list(G.out_edges(node2[0], data=True))):
                # print("edges are close")
                # print(
                #     f'{np.abs(edge1[2]["angle"] - edge2[2]["angle"])} = {edge1[2]["angle"]}, {edge2[2]["angle"]}'
                # )
                if np.isclose(normalize_angle(edge1[2]["angle"]),
                              normalize_angle(edge2[2]["angle"]),
                              atol=angle_threshold):
                    # print("removed node")
                    G, rem_1, rem_2 = merge_nodes(G, node1[0], node2[0])
                    nodes_to_remove.append(rem_1)
                    nodes_to_remove.append(rem_2)
                    skip = True
                    break
            if skip:
                continue
            for edge1, edge2 in zip(list(G.in_edges(node1[0], data=True)),
                                    list(G.in_edges(node2[0], data=True))):
                # print("edges are close")
                # print(
                #     f'{np.abs(edge1[2]["angle"] - edge2[2]["angle"])} = {edge1[2]["angle"]}, {edge2[2]["angle"]}'
                # )

                if np.isclose(normalize_angle(edge1[2]["angle"]),
                              normalize_angle(edge2[2]["angle"]),
                              atol=angle_threshold):
                    # print("removed node")
                    G, rem_1, rem_2 = merge_nodes(G, node1[0], node2[0])
                    nodes_to_remove.append(rem_1)
                    nodes_to_remove.append(rem_2)
                    break

    for node in nodes_to_remove:
        try:
            G.remove_node(node)
        except:
            pass
            # print(f"node {node} already removed")

    # print(f"number of edges1: {len(G.edges)}")
    print(f"Instance: {isinstance(G, nx.DiGraph)}")
    G_removed_duplicate_edges = remove_duplicate_edges(copy.deepcopy(G))
    # print(f"number of edges2: {len(G.edges)}")

    print(f"Instance: {isinstance(G_removed_duplicate_edges, nx.DiGraph)}")

    G_removed_dead_ends = remove_nodes_with_zero_outdegree(copy.deepcopy(G_removed_duplicate_edges))

    print(f"Instance: {isinstance(G_removed_dead_ends, nx.DiGraph)}")

    # print()
    # print(f"Number of nodes before merging: {nodes_before_merging}")
    # print(f"Number of nodes after merging: {len(G.nodes)}")
    # print()
    # print(f"Number of edges before merging: {edges_before_merging}")
    # print(f"Number of edges after merging: {len(G.edges)}")
    # print()
    return G_removed_dead_ends
    # your graph should now have the nodes merged as per your criteria
