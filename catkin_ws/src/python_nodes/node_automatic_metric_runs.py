"""This node will start all the processes and kills them for a restart when the
map server published, that the map is fully explored or a 30min timer ran
out."""
# pylint: disable=import-error
import signal
import subprocess
import threading
import time

import rospy
from std_msgs.msg import Header


class ProcessManager:
    """The class that manages the process."""

    def __init__(self, timeout):

        self.processes = []
        self.timeout = timeout
        self.start_time = time.time()

    def add_process(self, name, command):
        """This function adds a new process to the list."""
        self.processes.append({"name": name, "command": f"{command}", "process": None})

    def start_processes(self):
        """This function starts the processes."""
        for process_info in self.processes:
            print(f"Starting {process_info['name']}...")
            # Split the command into a list
            cmd_list = [
                "/bin/bash",
                "/home/buechner/collaborative_panoptic_mapping/catkin_ws/src/python_nodes/run_ros_command.sh", # pylint: disable=line-too-long
            ] + process_info["command"].split()
            process_info["process"] = subprocess.Popen(cmd_list)  # pylint: disable=consider-using-with
            time.sleep(5)

    def stop_processes(self):
        """This function stops the processes."""
        for process_info in self.processes:
            if process_info["process"]:
                print(f"Stopping {process_info['name']}...")
                process_info["process"].send_signal(signal.SIGINT)
                process_info["process"].wait()  # Wait for the process to terminate
                process_info["process"] = None
                time.sleep(1)
        print("All processes stopped.")
        time.sleep(15)

    def restart_processes(self):
        """Restart all processes that are within the object."""
        self.stop_processes()
        self.start_processes()

    def check_timeout(self):
        """Let the processes run while the timer is still up, else restart
        them."""
        while True:
            if time.time() - self.start_time >= self.timeout:
                self.restart_processes()
                self.start_time = time.time()
            time.sleep(1)


manager = ProcessManager(timeout=60 * 60)

manager.add_process("server", "roslaunch -v hdl_graph_slam map_server.launch")
manager.add_process("agent0", "roslaunch hdl_graph_slam hdl1.launch agent_no:=0")
manager.add_process("agent1", "roslaunch hdl_graph_slam hdl1.launch agent_no:=1")
manager.add_process("agent2", "roslaunch hdl_graph_slam hdl1.launch agent_no:=2")

# Start a separate thread to check for timeout
timeout_thread = threading.Thread(target=manager.check_timeout)
timeout_thread.start()

# Start all processes
manager.start_processes()


def end_run_callback(_):
    """This message is sent by the map server when the map is explored."""
    print("Received end_run message. Restarting all processes.")
    manager.restart_processes()


rospy.init_node("process_manager_node")
rospy.Subscriber("/end_run", Header, end_run_callback)
rospy.spin()
