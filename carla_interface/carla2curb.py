"""Python program to access the carla simulation, reset the world, map and
actors, respawn actors in a defined way.
"""
import concurrent.futures

# pylint: disable=import-error, global-at-module-level, global-statement, global-variable-not-assigned, too-many-statements
import sys
import time
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation
import ros
from interface import (
    ActorCollection,
    AgentPose,
    CarlaPythonInterface,
    ExportParameters,
    LidarParameters,
    SimulationParameters,
)

"""In the next 3 variable collections, it is possible to 
set some relevant variables for the simulation. 
Once the variables are changed, the values are used 
on the next start of carla2curb.

Some of the variables are useful for debugging purposes, 
some are only to test edge cases or to perform any 
simulation scenario that was relevant for the metrics. 
"""
export_parameters = ExportParameters(
    rosbag_file=False,
    ros_node=True,
)

simulation_parameters = SimulationParameters(
    port=2000,
    delta_t=0.1,
    town="town02",
    rendering=True,
    # this is useful to keep the traffic constantly moving
    set_lights_to_green=True,
    # careful when setting 'predefined_routes', 
    # there must be a valid list of route points within the code to use this option
    predefined_routes=False, 
    # this is useful to see the new map for planning or setting new waypoins.
    only_spawn_map=False,
)

actor_parameters = ActorCollection(number_of_agents=3, number_of_vehicles=30, number_of_walkers=0)

carla_world = CarlaPythonInterface(
    export_parameters=export_parameters,
    simulation_parameters=simulation_parameters,
    actor_collection=actor_parameters,
    lidar_parameters=LidarParameters(
        obs_range=64,
        lidar_bin=0.125,
        d_behind=12,
        lidar_range=80,
        rot_freq=10,
        points_per_second=800000,
        semantic=True,
    ),
)
carla_world.reset()

# this delay time is necessairy to give the carla server enought 
# time to collect all LiDAR scans. For faster simulation environments
# this time can be reduced to increase the simulation speed, but 
# be careful, because if chosen too low, LiDAR scans might be incomplete. 
global DELAY_TIME
DELAY_TIME = 0.05

# this is the simulation time that is written into the rosbags or 
# into the messages of the publishing ros node. Starts normally with 0.0
global SIM_TIME
SIM_TIME = 0.0

# start value to enumerate the frames
global NUM_FRAMES
NUM_FRAMES = 0

# somehow it is necessary to reset the agents to
# autopilot. In some cases they don't accept
# the first autopilot-set when resetting the agents
for ego in carla_world.ego:
    ego.set_autopilot(True)

carla_world.get_obs()

print("Starting loop..")

if export_parameters.rosbag_file:
    global BAG, BAG_NAME
    BAG_NAME = f"cr_{time.strftime('%H%M%S')}.bag"
    BAG = ros.create_bag(BAG_NAME)

if export_parameters.ros_node:
    ros.create_node("carla_publisher")

# publish the ids of the agent vehicles, this information is received by some 
# metric generating scrips and some scripts that automate test runs
ros.publish_agent_information("/agent_information", carla_world.world.id, carla_world.ego, SIM_TIME)

# this visualization option is currently not available
def update_image(_: Any) -> None:
    """Function to update the displayed image according to the defined refresh
    interval."""
    update_step()

    #image = BirdViewProducer.as_rgb(carla_world.birdeye_view)
    #im0.set_array(image)


def carla_tick() -> None:
    """Function to abstract the object."""
    carla_world.world.tick()


def update_step() -> None:
    """This function is called to do the tick in the simulation and then
    collects all the data from there.

    after that the data is handles according to the export properties.
    """
    skip = False
    buffer_time = time.time()
    start_time = buffer_time

    # this is used to catch the case, when the carla
    # simulation server stops working
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            future = executor.submit(carla_tick)
            _ = future.result(timeout=5.0)
        except concurrent.futures.TimeoutError:
            # in case the server is not responding, restart the environment
            print("carla-tick didnt return.")
            carla_world.carla_reconnect()
            print(carla_world.world)

            actors = carla_world.world.get_actors()
            print(len(actors))
            for actor in actors:
                print("=")
                print(actor)

            traffic_manager = carla_world.client.get_trafficmanager()
            print(traffic_manager)
            # reset the traffic manager
            traffic_manager.set_synchronous_mode(False)
            traffic_manager.set_synchronous_mode(True)

            carla_tick()
            # skip the next interval of data
            skip = True

    if not skip:
        tick_time = time.time() - buffer_time
        buffer_time = time.time()

        # I observed, that if there is no delay, the lidar callbacks don't have
        # enough time to collect the data
        global DELAY_TIME
        time.sleep(DELAY_TIME)
        delay_time = time.time() - buffer_time
        buffer_time = time.time()

        clouds: List[List[Tuple[float, float, float, int, int, float, float, float, float]]]
        states: List[AgentPose]

        # get the data from the carla server
        states, clouds = carla_world.get_obs()
        obs_time = time.time() - buffer_time
        buffer_time = time.time()

        # print to check if all 3 agent clouds have the same size
        global SIM_TIME, NUM_FRAMES
        SIM_TIME += simulation_parameters.delta_t
        NUM_FRAMES += 1

        if export_parameters.rosbag_file:
            global BAG
            ros.write_rosbag(BAG, SIM_TIME, states, clouds, "/velodyne_points_raw_", "/pose_")

        if export_parameters.ros_node:
            ros.tick(SIM_TIME)
            for agent_no in range(actor_parameters.number_of_agents):
                # this publishes the current position of the agents for: 
                # - currently we depend on an initial fixed position 
                # - to calculate the localization error metric
                ros.publish_pose(
                    f"/pose_{agent_no}",
                    agent_no,
                    SIM_TIME,
                    states[agent_no],
                    "world",
                )

                # this publishes the pointclouds in a format similar to a real LiDAR device 
                # or the output of a panoptic LiDAR algorithm. 
                # To replace the interface output with a real panoptic LiDAR algorithm 
                # check the stucture that is published here and 
                # reproduce it within the panoptic module
                ros.publish_pointcloud(
                    f"/velodyne_points_raw_{agent_no}",
                    SIM_TIME,
                    clouds[agent_no],
                    f"velodyne_{agent_no}",
                )

        export_time = time.time() - buffer_time

        # publish the ids of the agent vehicles
        ros.publish_agent_information("/agent_information", carla_world.world.id, carla_world.ego,
                                      SIM_TIME)
        # pylint: disable=line-too-long
        sys.stdout.write(
            f"""\rRecorded Time[min]: {(SIM_TIME / 60.0):.2f} / Frames: {NUM_FRAMES} / Dataset-Size: {len(clouds[0])} / Computation-time: {(time.time() - start_time):.3f}s / Tick-time: {tick_time:.3f}s / Delay: {delay_time:.3f}s / Obs-time: {obs_time:.3f}s / Export-time: {export_time:.3f}s """
        )
        sys.stdout.flush()
    else:
        print("skipped")

try:
    while True:
        update_step()

except KeyboardInterrupt:
    if export_parameters.rosbag_file:
        BAG.close()
        print(f"\nBag closed.\nName: {BAG_NAME}")
    print("Terminated.")
