"""This script prints the spawn points of the chosen map with their indices
into the map.

then a path can be created by using them to query the spawn point array
to get the actual coordinates.
"""
# pylint: disable=duplicate-code, invalid-name, import-error
import carla

load_world = True

# Connect to the Carla Simulator
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Load the "Town02" map
if load_world:
    world = client.load_world('town01')
else:
    world = client.get_world()

# exit(0)

spectator = world.get_spectator()
ego_transform = carla.Transform()
ego_transform.location.x = 100.0
ego_transform.location.y = 210.0
ego_transform.location.z = 250.0
ego_transform.rotation.pitch = -90
ego_transform.rotation.yaw = 90
spectator.set_transform(ego_transform)

# Get the blueprint library
blueprint_library = world.get_blueprint_library()

# Get all the possible spawn points of vehicles in the map
spawn_points = world.get_map().get_spawn_points()

for i, p in enumerate(spawn_points):
    world.debug.draw_string(p.location, str(i), life_time=300, color=carla.Color(255, 0, 0))
