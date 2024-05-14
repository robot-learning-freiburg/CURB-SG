"""Interface that implements the functions to interact with the carla
simulator."""
# pylint: disable=import-error, no-member, duplicate-code, too-few-public-methods, too-many-locals, too-many-statements
from __future__ import division

import random
import sys
import time
from typing import Any, Dict, List, NamedTuple, Tuple

import carla
import numpy as np
import numpy.typing as npt
from geometry_msgs.msg import Pose


class AgentPose(NamedTuple):
    """This Class is used so the main file does not need to import ros
    libraries."""
    pose: Pose


class ActorCollection(NamedTuple):
    """Collection of Actor attributes that are given to define the
    simulation."""

    number_of_vehicles: int
    number_of_agents: int
    number_of_walkers: int


class SimulationParameters(NamedTuple):
    """Parameter for the carla simulation."""

    port: int
    town: str
    delta_t: float
    rendering: bool
    set_lights_to_green: bool
    predefined_routes: bool
    only_spawn_map: bool


class ExportParameters(NamedTuple):
    """Parameter for the export settings."""

    rosbag_file: bool
    ros_node: bool


class LidarParameters(NamedTuple):
    """Needed parameters to define the used lidar sensor."""

    obs_range: int
    lidar_bin: float
    d_behind: int
    lidar_range: int
    rot_freq: int
    points_per_second: int
    semantic: bool


class AgentPointclouds(List[List[Tuple[float, float, float, int, int, float, float, float,
                                       float]]]):
    """Class to define the return of the observation function."""


class AgentStates(List[Pose]):
    """Class to define the return of the observation function."""


# pylint: disable=too-many-instance-attributes
def _set_carla_transform(pose: Any) -> carla.Transform:
    """Transforms a pose into a carla transform."""
    transform = carla.Transform()
    transform.location.x = pose[0]
    transform.location.y = pose[1]
    transform.rotation.yaw = pose[2]
    return transform


class CarlaPythonInterface:
    """This is the interface to the carla simulator."""

    def __init__(
        self,
        export_parameters: ExportParameters,
        simulation_parameters: SimulationParameters,
        actor_collection: ActorCollection,
        lidar_parameters: LidarParameters,
    ):
        self.export_parameters = export_parameters
        self.delta_t = simulation_parameters.delta_t
        self.rendering = simulation_parameters.rendering
        self.set_lights_to_green = simulation_parameters.set_lights_to_green
        self.predefined_routes = simulation_parameters.predefined_routes
        self.carla_port = simulation_parameters.port
        self.lidar_parameters = lidar_parameters
        self.actors = actor_collection

        self.ego: List[carla.Actor] = [None] * self.actors.number_of_agents

        self.lidar_data: List[Any] = [None] * self.actors.number_of_agents
        self.lidar_sensor: List[Any] = [None] * self.actors.number_of_agents
        for agent_no in range(self.actors.number_of_agents):
            self.lidar_sensor[agent_no]: carla.Actor = None  # type: ignore
            self.lidar_data[agent_no]: List[Any] = []  # type: ignore

        self.vehicle_polygons: List[Any] = []
        self.walker_polygons: List[Any] = []

        # Connect to carla server and get world object
        print("connecting to Carla server...")
        self.client = carla.Client("localhost", self.carla_port)
        self.client.set_timeout(10.0)

        # pylint: disable=bare-except
        try:
            print(self.client.get_world())
        except RuntimeError:
            print(f"No Server found at port:{simulation_parameters.port}")
            sys.exit(1)

        print("Carla server connected!")

        self.world = self.client.load_world(simulation_parameters.town)

        if (simulation_parameters.town.casefold()
                not in self.client.get_world().get_map().name.casefold()):
            print("Loading new map..")
            self.world = self.client.load_world(simulation_parameters.town)
        else:
            print("Using same map as before.")
            self.world = self.client.get_world()

        if simulation_parameters.only_spawn_map:
            self.set_spectator_position(apply_rotation=False)
            sys.exit(0)

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Get spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for _ in range(self.actors.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # Create the ego vehicle blueprint
        # index 3 has the special property that no part of the
        # own vehicle is captured on the lidar scans
        self.ego_bp = self.world.get_blueprint_library().filter("*vehicle*")[3]
        # self.ego_bp.set_attribute("role_name", "hero")

        # Lidar sensor
        self.lidar_height = 2.8
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))

        if self.lidar_parameters.semantic:
            self.lidar_bp = self.world.get_blueprint_library().find(
                "sensor.lidar.ray_cast_semantic")
        else:
            self.lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        self.lidar_bp.set_attribute("channels", str(self.lidar_parameters.obs_range))
        self.lidar_bp.set_attribute("range", str(self.lidar_parameters.lidar_range))
        self.lidar_bp.set_attribute("rotation_frequency", str(self.lidar_parameters.rot_freq))
        self.lidar_bp.set_attribute("points_per_second",
                                    str(self.lidar_parameters.points_per_second))
        self.lidar_bp.set_attribute("upper_fov", str(2))
        self.lidar_bp.set_attribute("lower_fov", str(-24.8))

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0
        self.time_step = 0

        self.all_actors: List[Any] = []

    def carla_reconnect(self) -> None:
        """Resets the objects and reconnects to the carla server."""
        # Connect to carla server and get world object
        print(self.client.get_world())
        self.world = None
        self.client = None

        print("connecting to Carla server...")
        self.client = carla.Client("localhost", self.carla_port)
        self.client.set_timeout(10.0)

        # pylint: disable=bare-except
        print(self.client.get_world())
        print("Carla server connected!")

        self.world = self.client.get_world()
        print("got new world")

    # pylint: disable=too-many-branches
    def reset(self) -> Tuple[Any, Any]:
        """This function clears all actors in the map, then respawns all actors
        and sensors again."""

        self.set_spectator_position(apply_rotation=False)

        # Delete sensors, vehicles and walkers
        self._clear_all_actors([
            "sensor.lidar.ray_cast_semantic",
            "vehicle.*",
            "controller.ai.walker",
            "walker.*",
        ])

        # # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons("vehicle.*")
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons("walker.*")
        self.walker_polygons.append(walker_poly_dict)

        # yapf: disable
        # if self.predefined_routes:
        # route_ind=[
        # [62, 79, 91, 80, 85, 62, 79, 91, 80, 85, 62, 79, 91, 80, 85, 62, ],
        # [61, 83, 90, 75, 61, 83, 90, 75, 61, 83, 90, 75, 61, 83, 90, 75, ]
        # ]

        # agent_spawn_point = [19, 93]
        # route_ind = [
        #     [28, 98, 42, 33, 10, 88, 28, 78, 61, 59, 11, 53, 90, 74, 43, 45, 4, 20, 14, 69, 62, ],
        #     [71, 46, 48, 10, 69, 60, 79, 74, 99, 65, 61, 59, 11, 53, 90, 75, 66, 1, 42, 33, 6]
        # ]

        if self.predefined_routes:
            agent_spawn_point = [88, 71]
            route_ind = [
                # [26, 78, 98, 42, 36, 10, 69, 62, 79, 74, 94, 10, 17, 26, 98, 94, 10, 17, 26, 98],
                [26, 78, 98, 42, 36, 10, 69, 62, 79, 50, 48, 10, 69, 62,  79, 74, 94],
                # [50, 48, 10, 69, 3, 71, 77, 61, 59, 11, 95, 93, 71, 75, 61, 59, 15, 96]
            ]
            # yapf: enable

            spawn_points = self.world.get_map().get_spawn_points()

            route_loc = []
            for agent_no, agent_route_ind in enumerate(route_ind):
                agent_loc_route = []
                for i, ind in enumerate(agent_route_ind):
                    agent_loc_route.append(spawn_points[ind].location)
                route_loc.append(agent_loc_route)

        traffic_manager = self.client.get_trafficmanager()

        for i in range(self.actors.number_of_agents):
            spawned = False
            while spawned is False:
                if self.predefined_routes:
                    transform = spawn_points[agent_spawn_point[i]]
                else:
                    transform = random.choice(self.vehicle_spawn_points)  # random
                spawned = self._try_spawn_ego_vehicle_at(transform, i)
                if spawned:
                    # traffic_manager.auto_lane_change(self.ego[i], False)

                    if self.predefined_routes:
                        traffic_manager.set_path(self.ego[i], route_loc[i])

        print(f"{self.actors.number_of_agents} Agents spawned.")

        # Spawn surrounding vehicles
        self.spawn_surrounding_vehicles()

        # Spawn pedestrians
        self.spawn_surrounding_walkers()

        self.all_actors = self.world.get_actors().filter("vehicle.*")

        time.sleep(self.delta_t)

        # Add lidar sensor
        def get_lidar_data(data: Any, index: int) -> None:
            """This is the function to react on the callback of the lidar
            sensor."""
            self.lidar_data[index] = data

        for agent_no in range(self.actors.number_of_agents):
            self.lidar_sensor[agent_no] = self.world.spawn_actor(self.lidar_bp,
                                                                 self.lidar_trans,
                                                                 attach_to=self.ego[agent_no])

        # sadly this is the only way to register those callback functions.
        # every time I tried to assign the callback with the loop variable agent_no
        # the registration did not work properly and took somehow only the last registered sensor
        for agent_no in range(self.actors.number_of_agents):
            if agent_no == 0:
                self.lidar_sensor[agent_no].listen(lambda data: get_lidar_data(data, 0))
            elif agent_no == 1:
                self.lidar_sensor[agent_no].listen(lambda data: get_lidar_data(data, 1))
            elif agent_no == 2:
                self.lidar_sensor[agent_no].listen(lambda data: get_lidar_data(data, 2))
            elif agent_no == 3:
                self.lidar_sensor[agent_no].listen(lambda data: get_lidar_data(data, 3))
            elif agent_no == 4:
                self.lidar_sensor[agent_no].listen(lambda data: get_lidar_data(data, 4))
            elif agent_no == 5:
                self.lidar_sensor[agent_no].listen(lambda data: get_lidar_data(data, 5))
            elif agent_no == 6:
                self.lidar_sensor[agent_no].listen(lambda data: get_lidar_data(data, 6))
            elif agent_no == 7:
                self.lidar_sensor[agent_no].listen(lambda data: get_lidar_data(data, 7))
            elif agent_no == 8:
                self.lidar_sensor[agent_no].listen(lambda data: get_lidar_data(data, 8))
            elif agent_no == 9:
                self.lidar_sensor[agent_no].listen(lambda data: get_lidar_data(data, 9))
            else:
                raise NotImplementedError("Only max 10 agents supported")

        if self.set_lights_to_green:
            list_actor = self.world.get_actors()
            for actor_ in list_actor:
                if isinstance(actor_, carla.TrafficLight):
                    # for any light, first set the light state, then set time. for yellow it is
                    # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
                    actor_.set_state(carla.TrafficLightState.Green)
                    actor_.set_green_time(100000.0)

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        print("Reset done.")

        # # Set the simulation to sync mode
        # init_settings = self.world.get_settings()
        # settings = self.world.get_settings()
        # settings.synchronous_mode = True
        # After that, set the TM to sync mode

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta_t
        settings.no_rendering_mode = not self.rendering
        self.world.apply_settings(settings)

        return self.get_obs()

    def set_spectator_position(self, apply_rotation: bool = False) -> None:
        """This is a function to set the spectator position to a desired one.

        This is important as it is quite difficult to move the specator
        when the simulation is running on a slow machine.
        """
        spectator = self.world.get_spectator()

        ego_transform = carla.Transform()
        ego_transform.location.x = 100.0
        ego_transform.location.y = 210.0
        ego_transform.location.z = 200.0

        if apply_rotation:
            ego_transform.rotation.pitch = -90
            ego_transform.rotation.yaw = 90
        spectator.set_transform(ego_transform)

    def spawn_surrounding_walkers(self) -> None:
        """This function controls the spawning of the walkers."""
        random.shuffle(self.walker_spawn_points)
        count = self.actors.number_of_walkers
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                count -= 1

    def spawn_surrounding_vehicles(self) -> None:
        """This function controls the spawning of the vehicles."""
        random.shuffle(self.vehicle_spawn_points)
        count = self.actors.number_of_vehicles
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, ):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), ):
                count -= 1

    def _try_spawn_random_vehicle_at(self, transform: carla.Transform) -> bool:
        """Try to spawn a surrounding vehicle at specific transform with random
        bluprint.

        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        blueprint = random.choice(self.world.get_blueprint_library().filter("vehicle.*"))
        blueprint.set_attribute("role_name", "autopilot")

        # there are models of the carlamotors type, that are very big and they are often not able to
        # to get the narrow curves in the town02. They stuck and the whole simulation is blocked.
        # so we do not spawn vehicles of this type.
        while blueprint.tags[0] == "carlamotors" or blueprint.tags[
                1] == "carlamotors" or blueprint.tags[1] == "mitsubishi" or blueprint.tags[
                    1] == "tesla" or blueprint.tags[1] == "ford":
            blueprint = random.choice(self.world.get_blueprint_library().filter("vehicle.*"))
        # print(blueprint)
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            time.sleep(0.1)
            vehicle.set_autopilot(True)
            return True
        return False

    def _try_spawn_random_walker_at(self, transform: carla.Transform) -> bool:
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter("walker.*"))
        # set as not invencible
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find("controller.ai.walker")
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp,
                                                             carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(
                1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform: carla.Transform, ego_index: int) -> bool:
        """Try to spawn the ego vehicle at specific transform.

        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for _, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            overlap = True
            break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego[ego_index] = vehicle
            print(f"New agent spawned: {vehicle}")
            # self.ego[ego_index].set_autopilot(True)
            return True
        return False

    def _get_actor_polygons(self, actor_filter: str) -> Dict[int, npt.NDArray[np.single]]:
        """Get the bounding box polygon of actors.

        Args:
          actor_filter: the filter indicating what type of actors we'll look at.
        Returns:
          actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(actor_filter):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x_val = trans.location.x
            y_val = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bounding_box = actor.bounding_box
            vec_x = bounding_box.extent.x
            vec_y = bounding_box.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[vec_x, vec_y], [vec_x, -vec_y], [-vec_x, -vec_y],
                                   [-vec_x, vec_y]]).transpose()
            # Get rotation matrix to transform to global coordinate
            rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(rot_matrix, poly_local).transpose() + np.repeat(
                [[x_val, y_val]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def get_obs(self) -> Tuple[AgentStates, AgentPointclouds]:
        """Get the observations.
        This is the function that is called regularly to get
        the current LiDAR data and the position of the agents."""

        for act in self.all_actors:
            online_actor = self.world.get_actor(act.id)
            if not online_actor.actor_state == carla.ActorState.Active:
                # Sometimes a carla actor stops working
                print("Actor Error!")
                print(f"id: {online_actor.id}")
                print(f"tags: {online_actor.type_id}")
                print(f"state: {online_actor.actor_state}")
                print("Removing actor from list..")

                self.all_actors = self.world.get_actors().filter("vehicle.*")

        # reset the data fields
        point_cloud: AgentPointclouds = [None] * self.actors.number_of_agents  # type: ignore
        state: AgentStates = [None] * self.actors.number_of_agents  # type: ignore

        # the lidar data of the agents is stored asynchronously in the self.lidar_data field
        # This function transforms the raw lidar data into the format that is sent out later
        for i in range(self.actors.number_of_agents):
            # Get point cloud data
            lidar = self.lidar_data[i]
            self.lidar_data[i] = []

            if lidar is not None:
                point_cloud[i] = [(location.point.x, -location.point.y, location.point.z,
                                   location.object_tag, location.object_idx, 0.5, 0.0, 0.0, 0.0)
                                  for location in lidar]

            # State observation of every agent
            if self.ego[i] is not None:
                ego_tf = self.ego[i].get_transform()

                pose = Pose()
                pose.position.x = ego_tf.location.x
                pose.position.y = -ego_tf.location.y
                pose.position.z = ego_tf.location.z

                # put the yaw angle in the z slot of the orientation
                # not very clean but sufficient for this case
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = -np.deg2rad(ego_tf.rotation.yaw)
                pose.orientation.w = 0.0

                state[i] = pose

        return state, point_cloud

    def _clear_all_actors(self, actor_filters: List[str]) -> None:
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == "controller.ai.walker":
                        actor.stop()
                    actor.destroy()
