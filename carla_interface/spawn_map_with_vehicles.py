"""This is a script to spawn multiple vehicles in a carla map to have a running
simulation.

this was used to record some videos for the presentation
"""
# pylint: disable=invalid-name, import-error
import random
import time

import carla

# Connect to the client and retrieve the world object
client = carla.Client("localhost", 2000)
world = client.load_world("Town02")
# world = client.get_world()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True  # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Set up the TM in synchronous mode
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# Set a seed so behaviour can be repeated if necessary
traffic_manager.set_random_device_seed(0)
random.seed(0)

# We will aslo set up the spectator so we can see what we do
spectator = world.get_spectator()

# Select some models from the blueprint library
models = [
    "dodge", "audi", "model3", "mini", "mustang", "lincoln", "prius", "nissan", "crown", "impala"
]
blueprints = []
for vehicle in world.get_blueprint_library().filter("*vehicle*"):
    if any(model in vehicle.id for model in models):
        blueprints.append(vehicle)

spawn_points = world.get_map().get_spawn_points()

# Set a max number of vehicles and prepare a list for those we spawn
max_vehicles = 50
max_vehicles = min([max_vehicles, len(spawn_points)])
vehicles = []

# Take a random sample of the spawn points and spawn some vehicles
for i, spawn_point in enumerate(random.sample(spawn_points, max_vehicles)):
    temp = world.try_spawn_actor(random.choice(blueprints), spawn_point)
    if temp is not None:
        vehicles.append(temp)

# Parse the list of spawned vehicles and give control to the TM through set_autopilot()
for vehicle in vehicles:
    vehicle.set_autopilot(True)
    # Randomly set the probability that a vehicle will ignore traffic lights
    traffic_manager.ignore_lights_percentage(vehicle, random.randint(0, 50))

print("Init done.we")

# Run the simulation so we can inspect the results with the spectator
while True:
    time.sleep(0.05)
    world.tick()
