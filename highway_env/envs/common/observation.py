from typing import List, Dict, TYPE_CHECKING, Optional
from gym import spaces
import numpy as np
import pandas as pd

from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.controller import MDPVehicle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class ObservationType(object):
    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()


class GrayscaleObservation(ObservationType):

    """
    An observation class that collects directly what the simulator renders

    Also stacks the collected frames as in the nature DQN.
    Specific keys are expected in the configuration dictionary passed.

    Example of observation dictionary in the environment config:
        observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],  #weights for RGB conversion,
            "stack_size": 4,
            "observation_shape": (84, 84)
        }

    Also, the screen_height and screen_width of the environment should match the
    expected observation_shape.
    """

    def __init__(self, env: 'AbstractEnv', config: dict) -> None:
        self.env = env
        self.config = config
        self.observation_shape = config["observation_shape"]
        self.shape = self.observation_shape + (config["stack_size"], )
        self.state = np.zeros(self.shape)

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(shape=self.shape,
                              low=0, high=1,
                              dtype=np.float32)
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        new_obs = self._record_to_grayscale()
        new_obs = np.reshape(new_obs, self.observation_shape)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[:, :, -1] = new_obs
        return self.state

    def _record_to_grayscale(self) -> np.ndarray:
        raw_rgb = self.env.render('rgb_array')
        return np.dot(raw_rgb[..., :3], self.config['weights'])


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env: 'AbstractEnv', horizon: int = 10, **kwargs: dict) -> None:
        self.env = env
        self.horizon = horizon

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(shape=self.observe().shape, low=0, high=1, dtype=np.float32)
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros((3, 3, int(self.horizon * self.env.config["policy_frequency"])))
        grid = compute_ttc_grid(self.env, time_quantization=1/self.env.config["policy_frequency"], horizon=self.horizon)
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.env.vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.env.vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0:lf+1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_speeds = 3
        v0 = grid.shape[0] + self.env.vehicle.speed_index - obs_speeds // 2
        vf = grid.shape[0] + self.env.vehicle.speed_index + obs_speeds // 2
        clamped_grid = padded_grid[v0:vf + 1, :, :]
        return clamped_grid


class KinematicObservation(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = False,
                 observe_intentions: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        self.env = env
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-1, high=1, dtype=np.float32)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.env.vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
                "vy": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.env.vehicle.to_dict()])[self.features]
        # Add nearby traffic
        # sort = self.order == "sorted"
        close_vehicles = self.env.road.close_vehicles_to(self.env.vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind)
        if close_vehicles:
            origin = self.env.vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs


class OccupancyGridObservation(ObservationType):

    """Observe an occupancy grid of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'vx', 'vy']
    GRID_SIZE: List[List[float]] = [[-5.5*5, 5.5*5], [-5.5*5, 5.5*5]]
    GRID_STEP: List[int] = [5, 5]

    def __init__(self,
                 env: 'AbstractEnv',
                 features: Optional[List[str]] = None,
                 grid_size: Optional[List[List[float]]] = None,
                 grid_step: Optional[List[int]] = None,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        """
        self.env = env
        self.features = features if features is not None else self.FEATURES
        self.grid_size = np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        self.grid_step = np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        grid_shape = np.asarray(np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step), dtype=np.int)
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.grid.shape, low=-1, high=1, dtype=np.float32)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
                "vy": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # Add nearby traffic
            self.grid.fill(0)
            df = pd.DataFrame.from_records(
                [v.to_dict(self.env.vehicle) for v in self.env.road.vehicles])
            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                for _, vehicle in df.iterrows():
                    x, y = vehicle["x"], vehicle["y"]
                    # Recover unnormalized coordinates for cell index
                    if "x" in self.features_range:
                        x = utils.lmap(x, [-1, 1], [self.features_range["x"][0], self.features_range["x"][1]])
                    if "y" in self.features_range:
                        y = utils.lmap(y, [-1, 1], [self.features_range["y"][0], self.features_range["y"][1]])
                    cell = (int((x - self.grid_size[0, 0]) / self.grid_step[0]),
                            int((y - self.grid_size[1, 0]) / self.grid_step[1]))
                    if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                        self.grid[layer, cell[1], cell[0]] = vehicle[feature]
            # Clip
            obs = np.clip(self.grid, -1, 1)
            return obs


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env: 'AbstractEnv', scales: List[float], **kwargs: dict) -> None:
        self.scales = np.array(scales)
        super().__init__(env, **kwargs)

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
            ))
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        if not self.env.vehicle:
            return {
            "observation": np.zeros((len(self.features),)),
            "achieved_goal": np.zeros((len(self.features),)),
            "desired_goal": np.zeros((len(self.features),))
        }

        obs = np.ravel(pd.DataFrame.from_records([self.env.vehicle.to_dict()])[self.features])
        goal = np.ravel(pd.DataFrame.from_records([self.env.goal.to_dict()])[self.features])
        obs = {
            "observation": obs / self.scales,
            "achieved_goal": obs / self.scales,
            "desired_goal": goal / self.scales
        }
        return obs


class AttributesObservation(ObservationType):
    def __init__(self, env: 'AbstractEnv', attributes: List[str], **kwargs: dict) -> None:
        self.env = env
        self.attributes = attributes

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict({
                attribute: spaces.Box(-np.inf, np.inf, shape=obs[attribute].shape, dtype=np.float32)
                for attribute in self.attributes
            })
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        return {
            attribute: getattr(self.env, attribute) for attribute in self.attributes
        }

class MyObservation(ObservationType):

    """Observe a Lidar of nearby vehicles."""

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """

    FEATURES: List[str] = ['static', 'dynamic']
    GRID_SIZE: List[List[float]] = [[-5.5*5, 5.5*5], [-5.5*5, 5.5*5]]
    GRID_STEP: List[int] = [5, 5]
    ANGLE_LIMIT: List[float] = [0., 2*np.pi, np.deg2rad(1)]
    RANGE_LIMIT: List[float] = [0., 100.]

    def __init__(self,
                 env: 'AbstractEnv',
                 features: Optional[List[str]] = None,
                 grid_size: Optional[List[List[float]]] = None,
                 grid_step: Optional[List[int]] = None,
                 angle_limit: Optional[List[float]] = None,
                 range_limit: Optional[List[float]] = None,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        """
        self.env = env
        self.features = features if features is not None else self.FEATURES
        self.grid_size = np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        self.grid_step = np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        grid_shape = np.asarray(np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step), dtype=np.int)
        self.grid = np.zeros((len(self.features), *grid_shape))

        # Lidar limitation
        self.angle_limit = np.array(angle_limit) if angle_limit is not None else np.array(self.ANGLE_LIMIT)
        self.range_limit = np.array(range_limit) if range_limit is not None else np.array(self.RANGE_LIMIT)
        self.__lidar_init()

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.grid.shape, low=0, high=1, dtype=np.float32)

    def observe(self) -> np.ndarray:
        if not self.env.road:
            obs = np.zeros(self.space().shape)
        else:
            obs = self.__lidar()
            # obs = np.clip(self.grid, -1, 1)
        return obs

    def __lidar_init(self):
        angle_min, angle_max, angle_step_size = self.angle_limit

        range_shape = np.asarray(abs(angle_max - angle_min) / angle_step_size, dtype=int)
        self.range = np.zeros(range_shape, dtype=np.float32)
        self.lidar_endpoint = np.zeros_like(self.range)

        # data = {'angle': np.arange(angle_min, angle_max, angle_step_size)}
        # self.df_lidar = pd.DataFrame(data, columns=['angle', 'x_min', 'y_min', 'x_max', 'y_max', 'intersect', 'x_i', 'y_i', 'distance'])
        # self.df_lidar['intersect'] = False

    def __lidar(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        df_v = pd.DataFrame.from_records([v.to_dict(self.env.vehicle) for v in self.env.road.vehicles])

        df_v['distance'] = utils.distance([0., 0.],[df_v['x'], df_v['y']])

        df_v['x1'] = np.nan
        df_v['y1'] = np.nan
        df_v['x2'] = np.nan
        df_v['y2'] = np.nan
        df_v['x3'] = np.nan
        df_v['y3'] = np.nan
        df_v['x4'] = np.nan
        df_v['y4'] = np.nan

        for i in df_v.index:
            [x1, y1], [x2, y2], [x3, y3], [x4, y4] = utils.corner_point(([df_v['x'][i], df_v['y'][i]],
                                                                         self.LENGTH,
                                                                         self.WIDTH,
                                                                         [df_v['cos_h'][i], df_v['sin_h'][i]]))
            df_v['x1'][i] = x1
            df_v['y1'][i] = y1
            df_v['x2'][i] = x2
            df_v['y2'][i] = y2
            df_v['x3'][i] = x3
            df_v['y3'][i] = y3
            df_v['x4'][i] = x4
            df_v['y4'][i] = y4

        angle_min, angle_max, angle_step_size = self.angle_limit
        # data = {'angle': np.arange(angle_min, angle_max, angle_step_size)}
        data = {'angle': np.arange(angle_min, angle_max, angle_step_size) + np.arccos(df_v['cos_h'][0])}
        df_lidar = pd.DataFrame(data, columns=['angle', 'x_min', 'y_min', 'x_max', 'y_max', 'intersect', 'x_i', 'y_i', 'distance'])

        df_lidar['intersect'] = False
        df_lidar['x_i'] = 0.
        df_lidar['y_i'] = 0.
        df_lidar['distance'] = self.range_limit[1]
        df_lidar['x_min'] = self.range_limit[0]*np.cos(df_lidar['angle'])
        df_lidar['y_min'] = self.range_limit[0]*np.sin(df_lidar['angle'])
        df_lidar['x_max'] = self.range_limit[1]*np.cos(df_lidar['angle'])
        df_lidar['y_max'] = self.range_limit[1]*np.sin(df_lidar['angle'])

        print(df_lidar)

        for i in df_lidar.index:
            line_lidar = ([df_lidar['x_min'][i], df_lidar['y_min'][i]],
                          [df_lidar['x_max'][i], df_lidar['y_max'][i]])

            for j in df_v.index:
                if j == 0:
                    continue
                else:
                    line0 = ([df_v['x1'][j], df_v['y1'][j]], [df_v['x2'][j], df_v['y2'][j]])
                    line1 = ([df_v['x2'][j], df_v['y2'][j]], [df_v['x3'][j], df_v['y3'][j]])
                    line2 = ([df_v['x3'][j], df_v['y3'][j]], [df_v['x4'][j], df_v['y4'][j]])
                    line3 = ([df_v['x4'][j], df_v['y4'][j]], [df_v['x1'][j], df_v['y1'][j]])
                    lines = [line0, line1, line2, line3]

                    for line in lines:
                        intersect, [x_i, y_i] = utils.is_line_intersect(line_lidar, line)
                        if intersect:
                            if df_lidar['intersect'][i] == False:
                                df_lidar['intersect'][i] = True
                                df_lidar['x_i'][i] = x_i
                                df_lidar['y_i'][i] = y_i
                                df_lidar['distance'][i] = utils.distance([0., 0.],[x_i, y_i])
                            else:
                                if utils.distance([0., 0.],[x_i, y_i]) < df_lidar['distance'][i]:
                                    df_lidar['x_i'][i] = x_i
                                    df_lidar['y_i'][i] = y_i
                                    df_lidar['distance'][i] = utils.distance([0., 0.],[x_i, y_i])

        # print(df_v)
        # print(df_lidar)

        obs = df_lidar[['angle', 'distance']].to_numpy()

        return obs

def observation_factory(env: 'AbstractEnv', config: dict) -> ObservationType:
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif config["type"] == "MyObservation":
        return MyObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
