"""This module is a node, that reads the traffic sign data from the semantic
map and generates bounding boxes around the traffic signs by using a DBScan
clustering algorithm."""
# pylint: disable=duplicate-code, import-error, too-many-locals
import time
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import rospy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


class NodeCluster:
    """This class is the node that clusters the traffic sign point clouds and
    generates the bounding boxes for all signs."""

    def __init__(self) -> None:

        self.sub_agent_0 = rospy.Subscriber("/graph", PointCloud2, self.map_points_callback)
        self.pub = rospy.Publisher("/feature_bounding_boxes", BoundingBoxArray, queue_size=10)

    def process_data(self, data: pc2) -> None:
        """Outsourced method."""

        pcl_signs: List[Tuple[float, ...]] = []
        pcl_poles: List[Tuple[float, ...]] = []

        for point in pc2.read_points(data, skip_nans=True):
            if int(point[6]) == 12:
                pcl_signs.append(point[:3])
            elif int(point[6]) == 5:
                pcl_poles.append(point[:3])

        clustered_bins_signs: List[List[Tuple[float, ...]]] = find_clusters(pcl_data=pcl_signs,
                                                                            eps=1.02,
                                                                            min_samples=25)
        print(f" Street-signs: {len(clustered_bins_signs)}")
        clustered_bins_poles: List[List[Tuple[float, ...]]] = find_clusters(pcl_data=pcl_poles,
                                                                            eps=1.02,
                                                                            min_samples=60)
        print(f" Poles:        {len(clustered_bins_poles)}")

        boxes: BoundingBoxArray = BoundingBoxArray()
        boxes.boxes = []

        for cluster in clustered_bins_signs:
            boxes.boxes.append(create_bounding_box(cluster, data.header.stamp, label=1))

        for cluster in clustered_bins_poles:
            boxes.boxes.append(create_bounding_box(cluster, data.header.stamp, label=2))

        boxes.header.frame_id = "world"
        boxes.header.stamp = data.header.stamp
        self.pub.publish(boxes)

    def map_points_callback(self, data: pc2) -> None:
        """This is the callback that gets the data and the index of the agent
        processes the data and then republishes again."""
        print()
        print(f"New map received at time: {data.header.stamp.to_sec():.2f}s")

        start_time = time.time()

        self.process_data(data)

        print(f"Calculation time: {((time.time() - start_time) * 1000):.0f}ms")

    def spin(self) -> None:
        """This in not really needed but to keep the node alive."""


def find_clusters(pcl_data: List[Tuple[float, ...]], eps,
                  min_samples) -> List[List[Tuple[float, ...]]]:
    """This method will do the clustering on the pointcloud of all street signs
    and return an ordered list of all clusters."""
    scaler: MinMaxScaler = MinMaxScaler()
    scaler.fit(pcl_data)

    global_scale = min(scaler.scale_[0], scaler.scale_[1])

    scaler.scale_ = (global_scale, global_scale, scaler.scale_[2])

    pcl_norm: List[Tuple[float, float]] = scaler.transform(pcl_data)
    scaled_eps = eps * global_scale

    dbscan: DBSCAN = DBSCAN(eps=scaled_eps, min_samples=min_samples)
    pcl_flat = [[pcl[0], pcl[1], 0.0] for pcl in pcl_norm]
    dbscan.fit_predict(pcl_flat)

    labels: npt.NDArray[np.intc] = dbscan.labels_
    clusters: int = len(set(labels)) - (1 if -1 in labels else 0)

    bins: List[List[Tuple[float, ...]]] = [[] for _ in range(clusters)]

    for lab, item in zip(labels, pcl_data):
        if int(lab) >= 0:
            bins[int(lab)].append(item)

    return bins


def create_bounding_box(pcl_poles: PointCloud2, stamp: Any, label=0) -> BoundingBox:
    """Returns a bounding box element that can be appended to a
    boundingboxarray."""
    box: BoundingBox = BoundingBox()
    box.header.stamp = stamp
    box.header.frame_id = "world"
    box.pose.orientation.w = 1

    pcl = np.asarray(pcl_poles)

    vec_min: Tuple[float, float, float] = np.min(pcl, axis=0)
    vec_max: Tuple[float, float, float] = np.max(pcl, axis=0)

    padding: float = 0.5

    x_min: float = vec_min[0] - padding
    x_max: float = vec_max[0] + padding
    y_min: float = vec_min[1] - padding
    y_max: float = vec_max[1] + padding
    z_min: float = vec_min[2] - padding
    z_max: float = vec_max[2] + padding

    x_pos: float = (x_max - x_min) / 2 + x_min
    y_pos: float = (y_max - y_min) / 2 + y_min
    z_pos: float = (z_max - z_min) / 2 + z_min

    box.pose.position.x = x_pos
    box.pose.position.y = y_pos
    box.pose.position.z = z_pos

    box.dimensions.x = x_max - x_min
    box.dimensions.y = y_max - y_min
    box.dimensions.z = z_max - z_min

    box.label = label

    return box


def main() -> None:
    """Init the node, and start it."""
    rospy.init_node("cluster_node")
    node = NodeCluster()

    print("Init done.")

    # pylint: disable=duplicate-code
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()


if __name__ == "__main__":
    main()
