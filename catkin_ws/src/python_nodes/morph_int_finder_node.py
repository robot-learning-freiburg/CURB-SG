"""
Intersection finder using a morphological skeletonization approach based on the road surface
"""
# pylint: skip-file
# pylint: disable=duplicate-code, import-error, no-name-in-module, inconsistent-quotes, invalid-name
import cv2
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from typing import List, Any

import rospy
import sknw
from abstract_graph import BoundingBoxCollection
from jsk_recognition_msgs.msg import BoundingBoxArray
from node_lane_graph import visualize_bounding_boxes, visualize_nodes
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from skimage.morphology import medial_axis
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray


def dbscan_filter(intersection_nodes, eps, min_samples):
    positions = np.array([(obj[0], obj[1]) for obj in intersection_nodes])
    print(f"pre  dbscan: {len(intersection_nodes)}")
    if len(intersection_nodes) == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)

    pre_filtered_observations = []
    # Get cluster labels
    labels = clustering.labels_

    # Handle noise points (label = -1)
    noise_points = np.array(positions)[labels == -1]
    pre_filtered_observations.extend(noise_points.tolist())

    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    # Handle clusters
    for label in unique_labels:
        cluster_points = np.array(positions)[labels == label]

        if len(cluster_points) >= 2:
            # Calculate the position in the middle of the cluster
            middle_point = np.mean(cluster_points, axis=0)
            pre_filtered_observations.append(middle_point.tolist())

    print(f"post dbscan: {len(pre_filtered_observations)}")
    return pre_filtered_observations

class MorphIntersectionFinder:
    """
    this class is the node that retrieves the road surface and finds intersections
    using morphological analysis.
    """

    def __init__(self) -> None:

        self.working = False
        self.last_stamp = 0.0
        self.sequential_id = 0

        self.bb_height = 0.0

        self.sub_static_pc = rospy.Subscriber("/map_server/map_points", PointCloud2, self.raw_pcl_callback)
        self.int_nodes_pub = rospy.Publisher("/intersection_street_nodes", MarkerArray, queue_size=2)

        self.nodes_pub = rospy.Publisher("/intersection_street_nodes_morphology", MarkerArray, queue_size=2)
        self.bb_pub = rospy.Publisher("/node_bounding_boxes", BoundingBoxArray, queue_size=2)

    def spin(self) -> None:
        """
        this in not really needed but to keep the node alive
        """

    def extract_street_points(self, data: PointCloud2):
        points_list = []

        # Iterate over the PointCloud2 data
        for point in pc2.read_points(data, skip_nans=True):
            x, y, z = point[:3]
            semantic_class = point[6]

            # Filter points with semantic class 7 (street)
            if semantic_class == 7:
                points_list.append([x, y, z])

        points_array = np.array(points_list, dtype=np.float32)
        return points_array

    def raw_pcl_callback(self, data: PointCloud2) -> None:
        """
        this is the callback that gets the data, processes the data
        and then republishes again
        """

        print()
        print("new map received.")
        self.sequential_id += 1

        self.last_stamp = data.header.stamp

        street_points = self.extract_street_points(data)

        slack = 50

        # msg_intersections: MarkerArray
        surface_img, x_min, y_min = MorphIntersectionFinder.preprocess_pointcloud(street_points, slack)
        graph = MorphIntersectionFinder.get_skeleton_graph(surface_img)

        # filter graph and remove short edges
        edges_to_remove = []
        for edge in graph.edges():
            # compute len of edge
            edge_len = np.linalg.norm(graph.nodes[edge[1]]['pos'] - graph.nodes[edge[0]]['pos'])
            # get node degree of both nodes
            node_0_degree, node_1_degree = graph.degree(edge[0]), graph.degree(edge[1])
            # use 50 for now as long as we do not change the image size
            if edge_len < 50 and (node_0_degree == 1 or node_1_degree == 1):
                edges_to_remove.append(edge)
        for e in edges_to_remove:
            graph.remove_edge(e[0], e[1])

        # print(f"Intersections found: {graph.number_of_nodes()}")

        plt.clf()
        plt.imshow(surface_img)
        for edge in graph.edges():
            plt.arrow(graph.nodes[edge[0]]['pos'][0], graph.nodes[edge[0]]['pos'][1],
                      graph.nodes[edge[1]]['pos'][0] - graph.nodes[edge[0]]['pos'][0],
                      graph.nodes[edge[1]]['pos'][1] - graph.nodes[edge[0]]['pos'][1],
                      head_width=1, head_length=2, fc='b', ec='b')
        # plt.show()
        plt.savefig("intersections.png")

        # filter graph for short edges without

        intersections: List[Any] = []
        streets: List[Any] = []

        intersection_positions: List[Any] = []
        for n in graph.nodes():
            if graph.degree(n) > 2:
                # Transform back to original coordinates
                pos_map = (graph.nodes[n]['pos'] / 2) - np.array([slack, slack]) - np.array([x_min, y_min])
                # node_position = pos_map
                intersection_positions.append(pos_map)

        filtered_positions = dbscan_filter(intersection_nodes=intersection_positions, eps=15.0, min_samples=2)

        for index, position in enumerate(filtered_positions):
            node_bb = BoundingBoxCollection(x_min=position[0] - 7.0,
                                            y_min=position[1] - 7.0,
                                            x_max=-position[0] + 7.0,
                                            y_max=-position[1] + 7.0)

            intersections.append({
                "label": index,
                "pos": position,
                "bounding_box": node_bb
            })

        marker = Marker()
        marker.action = Marker.DELETEALL
        delete_all_marker_array = MarkerArray()
        delete_all_marker_array.markers.append(marker)
        self.nodes_pub.publish(delete_all_marker_array)

        visualize_nodes(
            intersection_list=intersections,
            street_list=streets,
            publisher=self.nodes_pub,
            last_stamp=self.last_stamp,
            size=4.0,
            # height_offset=self.nodes_height_offset)
            height_offset=80.0)

        # visualize_bounding_boxes(intersection_bounding_boxes=intersections,
        #                          street_bounding_boxes=streets,
        #                          pub=self.bb_pub, last_stamp=self.last_stamp,
        #                          sequential_id=self.sequential_id,
        #                          height_offset=self.bb_height)
        print("intersections published.")


    @staticmethod
    def get_skeleton_graph(surface, threshold=0.05):
        skel, _ = medial_axis((surface > threshold).astype(np.uint8), return_distance=True)
        skel = skel.astype(np.uint8)

        # Remove border pixels
        skel[0, :] = 0
        skel[-1, :] = 0
        skel[:, 0] = 0
        skel[:, -1] = 0

        # Get sparse networkx graph
        sknw_graph = sknw.build_sknw(skel)

        # Convert to our graph format
        nx_graph = nx.DiGraph()
        for n, d in sknw_graph.nodes(data=True):
            pos = np.array([d['o'][1], d['o'][0]])
            nx_graph.add_node(n, pos=pos, score=1)
        for e in sknw_graph.edges():
            nx_graph.add_edge(e[0], e[1], weight=1)

        return nx_graph

    @staticmethod
    def preprocess_pointcloud(data: PointCloud2, slack: int):

        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(
        #     ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data))
        # point_cloud = np.asarray(data.points)

        # point_cloud = ros_numpy.numpify(data)
        surface = data[:, 0:2]
        x_extent = np.max(surface[:, 0]) - np.min(surface[:, 0])
        y_extent = np.max(surface[:, 1]) - np.min(surface[:, 1])

        img_extent = np.max([x_extent + slack, y_extent + slack]).astype(np.int32)
        x_min = np.abs(np.min(surface[:, 0]))
        y_min = np.abs(np.min(surface[:, 1]))

        img = np.zeros((img_extent * 2, img_extent * 2, 1), np.uint8)
        # cv2 draw point for all points in surface
        for point in surface:
            point += np.array([np.abs(np.min(surface[:, 0])), np.abs(np.min(surface[:, 1]))])
            point += np.array([slack, slack])
            point = point * 2
            # print((int(point[0]), int(point[1])))
            cv2.circle(img, (int(point[0]), int(point[1])), 2, 255, -1)

        # eroding
        # kernel = np.ones((5, 5), np.uint8)
        # img = cv2.erode(img, kernel)

        # smoothing
        kernel = np.ones((5, 5), np.float32) / 30
        dst = cv2.filter2D(img, -1, kernel)

        # dilate
        kernel = np.ones((9, 9), np.uint8)
        dst = cv2.dilate(dst, kernel)

        dst[dst < 150] = 0
        dst[dst >= 150] = 255

        return dst, x_min, y_min


def main() -> None:
    """
    init the node, and start it
    """

    rospy.init_node("morph_int_finder_node")
    node = MorphIntersectionFinder()

    print("Init done.")

    # pylint: disable=duplicate-code
    rate = rospy.Rate(2.0)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()


if __name__ == "__main__":
    main()
