"""This node subscribes to the topic, where the map server publishes the
current number of graph nodes, and the number of all created keyframes.

This data is then written into a csv file that can be plotted with
another python script.
"""
# pylint: disable=import-error
import csv

import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header


class MetricSubscriber:
    """This is the node that does, what I described in the module docstring."""

    def __init__(self):
        self.first_msg_received = False
        self.filename = None

        rospy.Subscriber("/merge_metric", PointStamped, self.callback)

        self.new_run_sub = rospy.Subscriber("/new_run", Header, self.new_run_callback)
        self.end_run_sub = rospy.Subscriber("/end_run", Header, self.end_run_callback)

        print("Init done.")

    def new_run_callback(self, data):
        """Callback for the new_run message published by the map server."""
        print("new run")
        print(data)

        self.first_msg_received = False
        self.filename = None

    def end_run_callback(self, data):
        """Callback for the end_run message published by the map server."""
        print("end run")
        print(data)

        self.first_msg_received = False
        self.filename = None

    def restart_run(self, timestamp):
        """Function to reset all variables to start a new metric run."""
        self.first_msg_received = True
        self.filename = f"metric_merge_{str(timestamp)}.csv"

        print(f"Writing data to: {self.filename}")
        with open(self.filename, "a", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "all_keyframes", "graph_nodes"])

    def callback(self, data):
        """Callback for the metric messages published by the map server."""
        if not self.first_msg_received:
            self.restart_run(data.header.stamp.to_sec())

        timestamp = f"{data.header.stamp.to_sec():.3f}"
        value_all_keyframes = data.point.x
        value_graph_nodes = data.point.y

        print(f"data: {timestamp}, {value_all_keyframes}, {value_graph_nodes}")

        with open(self.filename, "a", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, value_all_keyframes, value_graph_nodes])

    def spin(self):
        """Function needed for the spin mechanic."""
        rospy.spin()


def main():
    """Init the node, and start it."""
    rospy.init_node("merge_metric")
    node = MetricSubscriber()

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()


if __name__ == "__main__":
    main()
