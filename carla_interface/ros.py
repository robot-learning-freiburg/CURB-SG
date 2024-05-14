"""Class to handle the ros specific functionalities like writing a rosbag or
publishing the pointclouds by a node.

The generation of the message structure is similar for both tasks.
"""
# pylint: disable=unnecessary-list-index-lookup, import-error, too-many-arguments
from typing import Any, List, Tuple

import rosbag
import rospy
import tf
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped
from rosgraph_msgs.msg import Clock
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header


def write_rosbag(
    bag: rosbag.Bag,
    timestamp: float,
    states: List[Pose],
    clouds: List[List[Tuple[float, float, float, int, int, float]]],
    topic_clouds: str,
    topic_pose: str,
) -> None:
    """This function takes a pointcloud and exports it into an open rosbag,
    that will be closed when this python program is terminated."""

    print(f"length clouds: {len(clouds)}")

    for index, _ in enumerate(clouds):
        # Write pointcloud
        cloud_obj = create_pc2_message(clouds[index], timestamp, "global")
        bag.write(f"{topic_clouds}{index}", cloud_obj, cloud_obj.header.stamp)

        # write Pose
        pose_obj = create_pose_message(states[index], timestamp, "global")
        bag.write(f"{topic_pose}{index}", pose_obj, pose_obj.header.stamp)


def create_tf_transform_message(
    translation: Tuple[float, float, float],
    rotation: Tuple[float, ...],
    frame_id: str,
    child_frame_id: str,
    timestamp: float,
) -> TransformStamped:
    """Creates a stamped transform message from the given raw data."""
    msg = TransformStamped()
    msg.header = Header()
    msg.header.stamp = rospy.Time.from_sec(timestamp)
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.transform.translation.x = translation[0]
    msg.transform.translation.y = translation[1]
    msg.transform.translation.z = translation[2]
    msg.transform.rotation.x = rotation[0]
    msg.transform.rotation.y = rotation[1]
    msg.transform.rotation.z = rotation[2]
    msg.transform.rotation.w = rotation[3]
    return msg


def create_pose_message(state: Pose, timestamp: float, frame: str) -> PoseStamped:
    """Takes a pose and a timestamp and returns a ros pose message."""
    header = Header()
    header.stamp = rospy.Time.from_sec(timestamp)
    header.frame_id = frame
    pose_obj = PoseStamped()
    pose_obj.header = header
    pose_obj.pose = state
    return pose_obj


def create_pc2_message(
    cloud: List[Tuple[float, float, float, int, int, float]],
    timestamp: float,
    frame: str,
) -> Any:
    """Takes a pointcloud and a timestamp and returns a ros pc2 message."""
    header = Header()
    header.stamp = rospy.Time.from_sec(timestamp)
    header.frame_id = frame
    fields = [
        pc2.PointField("x", 0, pc2.PointField.FLOAT32, 1),
        pc2.PointField("y", 4, pc2.PointField.FLOAT32, 1),
        pc2.PointField("z", 8, pc2.PointField.FLOAT32, 1),
        pc2.PointField("intensity", 12, pc2.PointField.FLOAT32, 1),
        pc2.PointField("curvature", 16, pc2.PointField.FLOAT32, 1),
        pc2.PointField("idx", 20, pc2.PointField.FLOAT32, 1),
        pc2.PointField("normal_x", 24, pc2.PointField.FLOAT32, 1),
        pc2.PointField("normal_y", 28, pc2.PointField.FLOAT32, 1),
        pc2.PointField("normal_z", 32, pc2.PointField.FLOAT32, 1),
    ]

    cloud_obj = pc2.create_cloud(header, fields, cloud)
    cloud_obj.is_dense = True
    return cloud_obj


def publish_pointcloud(
    topic: str,
    timestamp: float,
    cloud: List[Tuple[float, float, float, int, int, float]],
    frame: str,
) -> None:
    """Simply publishes the given pcl on the give node."""
    cloud_obj = create_pc2_message(cloud, timestamp, frame)
    pub = rospy.Publisher(topic, PointCloud2, queue_size=10)
    pub.publish(cloud_obj)


def publish_tf(stamped_msg: Any) -> None:
    """Published a stamped transform to the tf topic."""
    pub = rospy.Publisher("/tf", TransformStamped, queue_size=10)
    pub.publish(stamped_msg)


def publish_pose(topic: str, agent_no: int, timestamp: float, pose: Pose, frame: str) -> None:
    """Simply publishes the given pose on the give node."""

    pose_msg = create_pose_message(pose, timestamp, frame)
    pub = rospy.Publisher(topic, PoseStamped, queue_size=10)
    pub.publish(pose_msg)

    tf_broadcaster = tf.TransformBroadcaster()
    tf_broadcaster.sendTransform(
        (pose.position.x, pose.position.y, 0.0),
        tf.transformations.quaternion_from_euler(0, 0, pose.orientation.z),
        rospy.Time.from_sec(timestamp),
        f"base_link_{agent_no}",
        "world",
    )


def publish_agent_information(topic: str, world_id: int, ego_list: List[Any],
                              timestamp: float) -> None:
    """This publishes the information like the agent number and the agent id
    for simplicity this is not wrapped into a custom message but just into a
    standard pose array message."""
    header = Header()
    header.stamp = rospy.Time.from_sec(timestamp)
    header.frame_id = str(world_id)

    msg = PoseArray()
    msg.header = header

    msg.poses = []
    for agent_no, agent in enumerate(ego_list):
        pose = Pose()
        pose.position.x = agent_no
        pose.position.y = agent.id
        msg.poses.append(pose)

    pub = rospy.Publisher(topic, PoseArray, queue_size=1)
    pub.publish(msg)


def tick(timestamp: float) -> None:
    """This publishes the simulation time as a ros clock server."""
    msg = Clock()
    msg.clock = rospy.Time.from_sec(timestamp)
    pub = rospy.Publisher("/clock", Clock, queue_size=1)
    pub.publish(msg)


def create_bag(file_name: str) -> rosbag.Bag:
    """Wrapper function to make the main file independent of ros libraries."""
    return rosbag.Bag(file_name, "w")


def create_node(name: str) -> None:
    """Creates the node, to make the main file independent of ros libraries."""
    rospy.init_node(name)
