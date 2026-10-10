from typing import Optional
from dataclasses import dataclass, field
from copy import deepcopy
import pandas as pd
import numpy as np
import functools

from constants import POSE_KEYPOINT_NAMES


class InvalidDataPoint(Exception):
    pass


@dataclass
class PlayerPosition:

    """
    Player position (meters) in a given frame
    """

    id: int
    position: tuple[float, float]

    def __post_init__(self):
        assert isinstance(self.position[0], float)
        assert isinstance(self.position[1], float)

    @property
    def key(self) -> str:
        return f"player{self.id}"

@dataclass
class BallPosition:

    """
    Ball position in a given frame

    The ball is tracked in the video frame, so `position` (pixels) is the only
    always-valid representation. `projection` applies the court floor-plane
    homography to it, which is physically meaningful only while the ball is on
    the ground (i.e. at a bounce) -- an airborne ball projects to where it would
    be if it were lying on the floor. Prefer `position` for trajectory work.

    Attributes:
        position: ball position (pixels) in the video frame
        visibility: 1 if the ball was detected in the given frame, else 0
        projection: ball position (meters) on the court floor plane
    """

    position: tuple[float, float]
    visibility: int
    projection: Optional[tuple[float, float]] = None


@dataclass
class PlayerPose:

    """
    Player pose keypoints in a given frame

    Keypoints are kept in video frame pixels on purpose: the homography maps the
    court floor plane, so it is only valid for points on the ground (a player's
    feet). Projecting a hand or head through it would place the joint at the
    wrong court position.

    Attributes:
        id: player id the pose was associated with
        keypoints: keypoint name -> position (pixels) in the video frame
    """

    id: int
    keypoints: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"player{self.id}"


@dataclass
class DataPoint:

    """
    Tracker objects data collected in a given frame

    Attributes:
        frame: frame of interest
        players_position: players position (meters) in the given frame
        ball_position: ball position in the given frame
        players_pose: players pose keypoints (pixels) in the given frame
    """

    frame: int = None
    players_position: list[PlayerPosition] = None
    ball_position: Optional[BallPosition] = None
    players_pose: Optional[list[PlayerPose]] = None

    def validate(self) -> None:

        if self.frame is None:
            raise InvalidDataPoint("Unknown frame")
        
        if self.players_position is None:
            print("data_analytics: WARNING(Missing players position)")
            return None
        
        players_ids = []
        for i, player_pos in enumerate(deepcopy(self.players_position)):
            player_id = player_pos.id

            if player_id in (1, 2, 3, 4):
                players_ids.append(player_id)
            else:
                del self.players_position[i]

        if len(players_ids) != len(set(players_ids)):
            raise InvalidDataPoint("N-plicate player id")
        
        if len(self.players_position) != 4:
            number_players_missing = 4 - len(self.players_position)
            print(f"{number_players_missing} player/s missing")
        
    def add_player_position(self, player_position: PlayerPosition):
        if self.players_position is None:
            self.players_position = [player_position]
        else:
            self.players_position.append(player_position)

    def add_ball_position(self, ball_position: BallPosition):
        self.ball_position = ball_position

    def add_player_pose(self, player_pose: PlayerPose):
        if self.players_pose is None:
            self.players_pose = [player_pose]
        else:
            self.players_pose.append(player_pose)

    def sort_players_position(self) -> Optional[list[PlayerPosition]]:
        if self.players_position:
            players_position = sorted(
                self.players_position, 
                key=lambda x: x.id,
            )
            return players_position
        
        print("data_analytics: impossible to sort, missing players position")
        return None

class DataAnalytics:

    """
    Tracker objects data collector 
    """

    def __init__(self, collect_pose: bool = True):
        self.collect_pose = collect_pose
        self.frames = [0]
        self.current_datapoint = DataPoint(frame=self.frames[-1])
        self.datapoints: list[DataPoint] = []

    def restart(self) -> None:
        self.__init__(collect_pose=self.collect_pose)

    @classmethod
    def from_dict(cls, data: dict):
        frames = data["frame"]
        instance = cls()
        instance.frames = frames

        datapoints = []
        for i in range(len(frames)):
            frame = frames[i]
            players_position = []
            for player_id in (1, 2, 3, 4):
                if (
                    data[f"player{player_id}_x"][i] is None
                    or 
                    data[f"player{player_id}_y"][i] is None
                ):
                    continue

                players_position.append(
                    PlayerPosition(
                        id=player_id,
                        position=(
                            data[f"player{player_id}_x"][i],
                            data[f"player{player_id}_y"][i],
                        )
                    )   
                )

            ball_position = None
            if data.get("ball_x", [None] * len(frames))[i] is not None:
                projection = None
                if data.get("ball_proj_x", [None] * len(frames))[i] is not None:
                    projection = (
                        data["ball_proj_x"][i],
                        data["ball_proj_y"][i],
                    )
                ball_position = BallPosition(
                    position=(data["ball_x"][i], data["ball_y"][i]),
                    visibility=data.get("ball_visibility", [0] * len(frames))[i],
                    projection=projection,
                )

            datapoints.append(
                DataPoint(
                    frame=frame,
                    players_position=players_position if players_position else None,
                    ball_position=ball_position,
                )
            )
        
        instance.datapoints = datapoints
        instance.current_datapoint = None

        return instance
    
    def into_dict(self) -> dict[str, list]:
        data = {
            "frame": [],
            "player1_x": [],
            "player1_y": [],
            "player2_x": [],
            "player2_y": [],
            "player3_x": [],
            "player3_y": [],
            "player4_x": [],
            "player4_y": [],
            # Ball position in video frame pixels, plus its floor-plane
            # projection in meters (only valid while the ball is grounded)
            "ball_x": [],
            "ball_y": [],
            "ball_visibility": [],
            "ball_proj_x": [],
            "ball_proj_y": [],
        }

        if self.collect_pose:
            # Pose keypoints stay in video frame pixels, see PlayerPose
            for player_id in (1, 2, 3, 4):
                for name in POSE_KEYPOINT_NAMES:
                    data[f"player{player_id}_{name}_x"] = []
                    data[f"player{player_id}_{name}_y"] = []

        for datapoint in self.datapoints:
            data["frame"].append(datapoint.frame)
            number_frames = len(data["frame"])

            players_position = datapoint.sort_players_position()
            if players_position:
                for player_position in players_position:
                    data[f"{player_position.key}_x"].append(
                        player_position.position[0]
                    )
                    data[f"{player_position.key}_y"].append(
                        player_position.position[1]
                    )

            ball_position = datapoint.ball_position
            if ball_position is not None:
                data["ball_x"].append(ball_position.position[0])
                data["ball_y"].append(ball_position.position[1])
                data["ball_visibility"].append(ball_position.visibility)
                if ball_position.projection is not None:
                    data["ball_proj_x"].append(ball_position.projection[0])
                    data["ball_proj_y"].append(ball_position.projection[1])
            else:
                data["ball_visibility"].append(0)

            if self.collect_pose and datapoint.players_pose:
                for player_pose in datapoint.players_pose:
                    for name, xy in player_pose.keypoints.items():
                        key_x = f"{player_pose.key}_{name}_x"
                        if key_x not in data:
                            continue
                        data[key_x].append(xy[0])
                        data[f"{player_pose.key}_{name}_y"].append(xy[1])

            # Append missing values
            for k, v in data.items():
                if len(v) < number_frames:
                    data[k].append(None)

        print("data_analytics: missing values")
        for k, v in data.items():
            number_missing = len([x for x in v if x is None])
            if number_missing:
                print(f"data_analytics: {k} - {number_missing}/{len(v)}")

        return data

    def __len__(self) -> int:
        return len(self.frames)

    def update(self):
        self.current_datapoint.validate()
        self.datapoints.append(self.current_datapoint)
        self.current_datapoint = DataPoint(frame=self.frames[-1])
    
    def step(self, x: int = 1) -> None:
        new_frame = self.frames[-1] + 1

        assert new_frame not in self.frames

        self.frames.append(new_frame)
        self.update()

    def add_player_position(
        self,
        id: int,
        position: tuple[float, float],
    ):
        self.current_datapoint.add_player_position(
            PlayerPosition(
                id=id,
                position=position,
            )
        )

    def add_ball_position(
        self,
        position: tuple[float, float],
        visibility: int,
        projection: Optional[tuple[float, float]] = None,
    ):
        self.current_datapoint.add_ball_position(
            BallPosition(
                position=position,
                visibility=visibility,
                projection=projection,
            )
        )

    def add_player_pose(
        self,
        id: int,
        keypoints: dict[str, tuple[float, float]],
    ):
        self.current_datapoint.add_player_pose(
            PlayerPose(
                id=id,
                keypoints=keypoints,
            )
        )

    def into_dataframe(self, fps: int) -> pd.DataFrame:
        """
        Retrieves a dataframe with additional features
        """

        def norm(x: float, y: float) -> float:
            return np.sqrt(x**2 + y**2)

        def calculate_distance(row, player_id: int):
            return norm(
                row[f"player{player_id}_deltax1"], 
                row[f"player{player_id}_deltay1"], 
            )
        
        def calculate_norm_velocity(row, player_id: int, frame_interval: int) -> float:
            return norm(
                row[f"player{player_id}_Vx{frame_interval}"],
                row[f"player{player_id}_Vy{frame_interval}"],
            )

        def calculate_norm_acceleration(row, player_id: int, frame_interval: int) -> float:
            return norm(
                row[f"player{player_id}_Ax{frame_interval}"],
                row[f"player{player_id}_Ay{frame_interval}"],
            )

        frame_intervals = (1, 2, 3, 4)
        player_ids = (1, 2, 3, 4)

        df = pd.DataFrame(self.into_dict())
        df["time"] = df["frame"] * (1/fps)

        for frame_interval in frame_intervals:
            # Time in seconds between each frame for a given frame interval
            df[f"delta_time{frame_interval}"] = df["time"].diff(frame_interval)
            for player_id in player_ids:
                for pos in ("x", "y"):
                    # Displacement in x and y for each of the players 
                    # for a given time interval
                    df[
                        f"player{player_id}_delta{pos}{frame_interval}"
                    ] = df[f"player{player_id}_{pos}"].diff(frame_interval)

                    # Velocity in x and y for each of the players 
                    # for a given time interval
                    eval_string_velocity = f"""
                    player{player_id}_delta{pos}{frame_interval} / delta_time{frame_interval}
                    """
                    df[f"player{player_id}_V{pos}{frame_interval}"] = df.eval(
                        eval_string_velocity,
                    )

                    # Velocity difference in x and y for each of the players 
                    # for a given time interval
                    df[
                        f"player{player_id}_deltaV{pos}{frame_interval}"
                    ] = df[f"player{player_id}_V{pos}{frame_interval}"].diff(frame_interval)

                    # Acceleration in x and y for each of the players
                    # for a given time interval
                    eval_string_acceleration = f"""
                    player{player_id}_deltaV{pos}{frame_interval} / delta_time{frame_interval}
                    """
                    df[f"player{player_id}_A{pos}{frame_interval}"] = df.eval(
                        eval_string_acceleration,
                    )
                
                # Calculate player distance in between frames
                df[f"player{player_id}_distance"] = df.apply(
                    functools.partial(calculate_distance, player_id=player_id),
                    axis=1,
                )

                # Calculate norm velocity for each of the players
                # for a given time interval
                df[f"player{player_id}_Vnorm{frame_interval}"] = df.apply(
                    functools.partial(
                        calculate_norm_velocity, 
                        player_id=player_id,
                        frame_interval=frame_interval,
                    ),
                    axis=1,
                )

                # Calculate norm acceleration for each of the players
                # for a given time interval
                df[f"player{player_id}_Anorm{frame_interval}"] = df.apply(
                    functools.partial(
                        calculate_norm_acceleration,
                        player_id=player_id,
                        frame_interval=frame_interval,
                    ),
                    axis=1,
                )

        # Ball trajectory features, in video frame pixels rather than meters:
        # the floor-plane projection is only valid while the ball is grounded,
        # see BallPosition. A racket contact reverses the ball direction, so
        # ball_turn_deg is the column hit detection keys on.
        df["ball_visibility"] = df["ball_visibility"].fillna(0).astype(int)

        # An undetected ball still carries a placeholder coordinate, so mask it
        # out to keep it from polluting the derived features
        is_visible = df["ball_visibility"] == 1
        ball_x = pd.to_numeric(df["ball_x"], errors="coerce").where(is_visible)
        ball_y = pd.to_numeric(df["ball_y"], errors="coerce").where(is_visible)

        ball_dx = ball_x.diff()
        ball_dy = ball_y.diff()
        ball_direction = np.degrees(np.arctan2(ball_dy, ball_dx))

        # Wrap the heading change into [0, 180] so a reversal reads as ~180
        # regardless of which way it crossed the discontinuity
        ball_turn = ball_direction.diff().abs()
        ball_turn = ball_turn.where(ball_turn <= 180, 360 - ball_turn)

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "ball_dx": ball_dx,
                        "ball_dy": ball_dy,
                        "ball_speed_px": np.sqrt(ball_dx**2 + ball_dy**2),
                        "ball_direction_deg": ball_direction,
                        "ball_turn_deg": ball_turn,
                    },
                    index=df.index,
                ),
            ],
            axis=1,
        )

        return df


        



        
    