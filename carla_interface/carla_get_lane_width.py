"""This is a helper script to read out the lane width 
at a specified location on the map."""
# pylint: skip-file
import math

import carla

client = carla.Client('localhost', 2000)
world = client.get_world()
map = world.get_map()

spawn_points = world.get_map().get_spawn_points()

# Get the location where you want to find the lane width
location = spawn_points[10].location

# Get the waypoint at this location for driving lanes
waypoint = map.get_waypoint(location)

print(f"lane width: {waypoint.lane_width}")
