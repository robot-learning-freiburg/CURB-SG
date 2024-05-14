"""This ros node will process the incoming pointclouds from the sensor and
simulate a semantic network that could differentiate between different classes.
the input are the pointclouds from the agents,

the output are 2 pointclouds per agent:
    - one containing static object
    - one containing dynamic objects.

The definition of the separation is contained withing a xml file
"""
# pylint: disable=duplicate-code, undefined-variable, import-error, too-many-arguments, too-many-locals
import sys
import time
from typing import Any, Dict, List, NamedTuple, Tuple

import carla
import numpy as np
import ros_numpy
import rospy
import tf
from geometry_msgs.msg import Point32, PolygonStamped, PoseArray
from jsk_recognition_msgs.msg import (
    BoundingBox,
    BoundingBoxArray,
    PolygonArray,
)
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from shapely.geometry import Point as sPoint
from shapely.geometry.polygon import Polygon as sPoly
from std_msgs.msg import Header


class ThingPolygon(NamedTuple):
    """Object for the combination of id and polygon."""

    id: int
    polygon: sPoly


class AgentInformation(NamedTuple):
    """Object for the agent describtion."""

    no: int
    id: int
    actor: carla.Actor


class NodeCarlaIdProvider:
    """This class is the node that is dividing the pointclouds."""

    def __init__(self) -> None:

        # Connect to carla server and get world object
        print("connecting to Carla server...")
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        # pylint: disable=bare-except
        try:
            print(client.get_world())
        except RuntimeError:
            print("No Server found at the given port.")
            sys.exit(1)

        self.world = client.get_world()

        print("Carla server connected!")

        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_raw_0", PointCloud2,
                                            self.raw_pcl_callback, 0)
        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_raw_1", PointCloud2,
                                            self.raw_pcl_callback, 1)
        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_raw_2", PointCloud2,
                                            self.raw_pcl_callback, 2)
        self.sub_agent_0 = rospy.Subscriber("/velodyne_points_raw_3", PointCloud2,
                                            self.raw_pcl_callback, 3)

        rospy.Subscriber("/agent_information", PoseArray, self.agent_information_callback)

        self.agent_information_received: bool = False
        self.agent_list: List[AgentInformation] = []
        self.polygon_list: List[ThingPolygon] = []
        self.raw_msg: Any = None

    def agent_information_callback(self, data: Any) -> None:
        """Callback method."""
        if not self.agent_information_received or self.world.id != int(data.header.frame_id):
            for agent_data in data.poses:
                actor = self.world.get_actor(int(agent_data.position.y))
                if actor is None:
                    raise CarlaAgentActorNotFound  # type: ignore
                self.agent_list.append(
                    AgentInformation(
                        no=int(agent_data.position.x),
                        id=int(agent_data.position.y),
                        actor=actor,
                    ))
                print(f"Agent updated {actor}")
            self.agent_information_received = True

    def process_data(self, data: pc2, agent_no: int) -> Any:
        """Outsourced method."""
        if not self.agent_information_received:
            print("Missing Agent Information.")
            return None

        # find an id within the pointcloud
        # is the id already in the dict?
        #
        # no:   get the id from carla for that point
        #       change the id to the one from carla
        #       add entry into a dictionary so future entries can use that id
        # yes:  change the id to the one from the dict
        #
        # return the pointcloud

        # alternative:
        # get the unique ids from the frame
        # get the carla ids for those unique ones
        # filter the pcl for the ids
        # change all the ids in the filtered pcl

        print()

        self.polygon_list = []

        # lidar = self.world.get_actors().filter("*lidar*")[0]
        pub_box = rospy.Publisher("/test_boxes", BoundingBoxArray, queue_size=10)
        pub_poly = rospy.Publisher("/test_ploy", PolygonArray, queue_size=10)

        header = Header()
        header.frame_id = "map"
        header.stamp = self.raw_msg.header.stamp

        boxes: BoundingBoxArray = BoundingBoxArray()
        boxes.boxes = []
        boxes.header = header

        polygons = PolygonArray()
        polygons.header = header
        polygons.polygons = []

        for vehicle in self.world.get_actors().filter("*vehicle*"):
            position = vehicle.get_transform().transform(vehicle.bounding_box.location)
            yaw = np.deg2rad(vehicle.get_transform().rotation.yaw)

            dimension: Tuple[float, float, float] = (
                vehicle.bounding_box.extent.x + 0.5,
                vehicle.bounding_box.extent.y + 0.5,
                vehicle.bounding_box.extent.z,
            )

            box = self.create_bounding_box(header, position, dimension, yaw)
            boxes.boxes.append(box)

            poly = self.create_polygon(position, dimension, header, yaw, vehicle.id)
            polygons.polygons.append(poly)

        pub_box.publish(boxes)
        pub_poly.publish(polygons)

        id_dict: Dict[int, int] = {}

        point_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(data)

        map_array = map(np.array, point_cloud)
        point_cloud = np.array(list(map_array))

        for point in point_cloud:
            if point[4] > 0 and point[4] not in id_dict and point[3] in (10, 100000):
                point_in_world = (self.agent_list[agent_no].actor.get_transform().transform(
                    carla.Location(point[0].item(), -point[1].item(), 1.0)))

                carla_id = self.get_carla_id(point_in_world)
                if carla_id is not None:
                    id_dict[point[4]] = carla_id

        print()
        print("results: ")
        print(id_dict)

        print(point_cloud.shape)

        for point in point_cloud:
            if point[4] in id_dict:
                point[4] = id_dict[point[4]]

        return point_cloud

    def get_carla_id(self, point: carla.Location) -> int:
        """This method finds the ground truth carla ID by finding the bounding
        box, that containts the given coordinate."""
        print()
        point2d = sPoint(point.x, point.y)
        for poly in self.polygon_list:

            if poly.polygon.contains(point2d):
                return poly.id

        raise CarlaIdNotFound  # type: ignore

    def create_polygon(
        self,
        position: carla.Location,
        dimension: Tuple[float, float, float],
        header: Any,
        yaw: float,
        poly_id: int,
    ) -> PolygonStamped:
        """Creates a polygon from the given position, the dimension and the yaw
        angle."""
        poly = PolygonStamped()
        poly.header = header

        raw_polygon = [
            self.rotate_point32(
                pos_x=position.x,
                dim_x=dimension[0],
                pos_y=position.y,
                dim_y=dimension[1],
                angle=yaw,
            ),
            self.rotate_point32(
                pos_x=position.x,
                dim_x=dimension[0],
                pos_y=position.y,
                dim_y=-dimension[1],
                angle=yaw,
            ),
            self.rotate_point32(
                pos_x=position.x,
                dim_x=-dimension[0],
                pos_y=position.y,
                dim_y=-dimension[1],
                angle=yaw,
            ),
            self.rotate_point32(
                pos_x=position.x,
                dim_x=-dimension[0],
                pos_y=position.y,
                dim_y=dimension[1],
                angle=yaw,
            ),
        ]

        shap_poly = sPoly([
            self.rotate_point(
                pos_x=position.x,
                dim_x=dimension[0],
                pos_y=position.y,
                dim_y=dimension[1],
                angle=yaw,
            ),
            self.rotate_point(
                pos_x=position.x,
                dim_x=dimension[0],
                pos_y=position.y,
                dim_y=-dimension[1],
                angle=yaw,
            ),
            self.rotate_point(
                pos_x=position.x,
                dim_x=-dimension[0],
                pos_y=position.y,
                dim_y=-dimension[1],
                angle=yaw,
            ),
            self.rotate_point(
                pos_x=position.x,
                dim_x=-dimension[0],
                pos_y=position.y,
                dim_y=dimension[1],
                angle=yaw,
            ),
        ])

        self.polygon_list.append(ThingPolygon(id=poly_id, polygon=shap_poly))

        poly.polygon.points = raw_polygon
        return poly

    def rotate_point32(self, pos_x: float, dim_x: float, pos_y: float, dim_y: float,
                       angle: float) -> Point32:
        """Rotates a points dimension around a given angle and returns a
        Point32 object."""
        return Point32(
            x=pos_x + (dim_x * np.cos(angle) - dim_y * np.sin(angle)),
            y=pos_y + (dim_y * np.cos(angle) - dim_x * np.sin(angle)),
            z=0.0,
        )

    def rotate_point(self, pos_x: float, dim_x: float, pos_y: float, dim_y: float,
                     angle: float) -> Tuple[float, float]:
        """Rotates a points dimension with a given angle."""
        return (
            pos_x + (dim_x * np.cos(angle) - dim_y * np.sin(angle)),
            pos_y + (dim_y * np.cos(angle) - dim_x * np.sin(angle)),
        )

    def create_bounding_box(
        self,
        header: Any,
        position: carla.Location,
        dimension: Tuple[float, float, float],
        yaw: float,
    ) -> BoundingBox:
        """Creates a bounding box with a given position, dimenstion and
        orientation."""
        box: BoundingBox = BoundingBox()

        box.header = header

        quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
        box.pose.orientation.x = quat[0]
        box.pose.orientation.y = quat[1]
        box.pose.orientation.z = quat[2]
        box.pose.orientation.w = quat[3]
        box.value = 1
        box.label = 2
        box.pose.position = position
        box.dimensions.x = dimension[0] * 2
        box.dimensions.y = dimension[1] * 2
        box.dimensions.z = dimension[2] * 2
        return box

    def publish_data(self, msg_dyn: PointCloud2, agent_no: int) -> None:
        """Outsourced method to keep thing separated."""

        # pub = rospy.Publisher(
        #     f"/velodyne_points_dyn_{agent_no}", PointCloud2, queue_size=10
        # )
        # pub.publish(msg_dyn)

    def raw_pcl_callback(self, data: PointCloud2, agent_no: int) -> None:
        """This is the callback that gets the data and the index of the agent
        processes the data and then republishes again."""
        start = time.time()
        # msg_dyn: PointCloud2
        # msg_static: PointCloud2
        self.raw_msg = data
        new_pcl: List[Tuple[float, float, float, int, int,
                            float]] = self.process_data(data, agent_no)

        cloud_obj = self.create_pc2_message(new_pcl, data.header.stamp, "map")
        pub = rospy.Publisher("/new_pcl", PointCloud2, queue_size=10)
        pub.publish(cloud_obj)

        # publish_data(msg_dyn, msg_static, agent_no=agent_no)
        print(f"Agent {agent_no} - Computation Time: {(time.time() - start):.3f}")

    def spin(self) -> None:
        """This in not really needed but to keep the node alive."""

    def create_pc2_message(
        self,
        cloud: List[Tuple[float, float, float, int, int, float]],
        timestamp: float,
        frame: str,
    ) -> Any:
        """Takes a pointcloud and a timestamp and returns a ros pc2 message."""
        header = Header()
        header.stamp = timestamp
        header.frame_id = frame
        fields = [
            pc2.PointField("x", 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField("y", 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField("z", 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField("intensity", 12, pc2.PointField.FLOAT32, 1),
            pc2.PointField("curvature", 16, pc2.PointField.FLOAT32, 1),
            pc2.PointField("idx", 20, pc2.PointField.FLOAT32, 1),
        ]

        cloud_obj = pc2.create_cloud(header, fields, cloud)
        cloud_obj.is_dense = True
        return cloud_obj


def main() -> None:
    """Init the node, and start it."""
    rospy.init_node("NodeCarlaIdProvider")
    node = NodeCarlaIdProvider()

    # pylint: disable=duplicate-code
    # rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        # rate.sleep()
        time.sleep(1.0)


if __name__ == "__main__":
    main()
