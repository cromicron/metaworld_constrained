from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld_constrained.envs.asset_path_utils import full_v2_path_for
from metaworld_constrained.envs.mujoco.sawyer_xyz.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld_constrained.envs.mujoco.utils import reward_utils
from metaworld_constrained.types import InitConfigDict


class SawyerDoorCloseEnvV2(SawyerXYZEnv):
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        constraint_mode: Literal["static", "relative", "absolute", "random"] = "relative",
        constraint_size: float = 0.03,
        include_const_in_obs: bool = True,
    ) -> None:
        goal_low = (0.2, 0.65, 0.1499)
        goal_high = (0.3, 0.75, 0.1501)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.85, 0.15)
        obj_high = (0.1, 0.95, 0.15)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            constraint_mode=constraint_mode,
            constraint_size=constraint_size,
            include_const_in_obs=include_const_in_obs,
        )

        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.1, 0.95, 0.15], dtype=np.float32),
            "hand_init_pos": np.array([-0.5, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.2, 0.8, 0.15])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.door_qpos_adr = self.model.joint("doorjoint").qposadr.item()
        self.door_qvel_adr = self.model.joint("doorjoint").dofadr.item()

        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_door_pull.xml")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.data.geom("handle").xpos.copy()

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return Rotation.from_matrix(
            self.data.geom("handle").xmat.reshape(3, 3)
        ).as_quat()

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        move_index_pos = 7 # due to the additional object in main-scene, indexes change
        move_index_vel = 6
        qpos[9 + move_index_pos: 12 + move_index_pos] = pos.copy()
        qvel[9 + move_index_vel: 15 + move_index_vel] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    def _calc_constraint_pos(self):
        if self._last_rand_vec.shape[0] == 6:
            pos_constraint = self._last_rand_vec[-3: ]
        elif self._constraint_mode == "relative":

            pos_constraint = (self._last_rand_vec[: 2]  - np.array([0, 0.2]) + self.hand_init_pos[: -1]) / 2
            pos_constraint = np.hstack((pos_constraint, [0.02]))

        elif self._constraint_mode == "random":
            # randomly place constraint in the box of object and goal
            # if the goal is not on the table, the constraint can be at the
            # same y-coordinate as the goal. If it's on the table, then
            # it could potentially overlap with the goal

            goal_pos = self._last_rand_vec[0: 3].copy()
            x_min = self.hand_init_pos[0] + 0.1
            y_min = goal_pos[1] - 0.35
            max_coord = goal_pos - np.array([0.3, 0.2, 0.02])

            pos_constraint = np.random.uniform(
                np.array([x_min, y_min, 0.02]),
                max_coord,
                size=3,
            )

        elif self._constraint_mode == "absolute":
            pos_constraint = np.array([0, 0.7, 0.02])
        else:
            pos_constraint = self.data.body("constraint_box").xipos
        return pos_constraint

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.objHeight = self.data.geom("handle").xpos[2]
        obj_pos = self._get_state_rand_vec()[:3]
        self.obj_init_pos = obj_pos
        goal_pos = obj_pos.copy() + np.array([0.2, -0.2, 0.0])
        self._target_pos = goal_pos

        self.model.body("door").pos = self.obj_init_pos
        self.model.site("goal").pos = self._target_pos

        # keep the door open after resetting initial positions
        self._set_obj_xyz(np.array(-1.5708))
        self.model.site("goal").pos = self._target_pos
        return self._get_obs()

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        reward, obj_to_target, in_place = self.compute_reward(action, obs)
        info = {
            "obj_to_target": obj_to_target,
            "in_place_reward": in_place,
            "success": float(obj_to_target <= 0.08),
            "near_object": 0.0,
            "grasp_success": 1.0,
            "grasp_reward": 1.0,
            "unscaled_reward": reward,
        }
        return reward, info

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float]:
        assert (
            self._target_pos is not None and self.hand_init_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        _TARGET_RADIUS: float = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        target = self._target_pos

        tcp_to_target = float(np.linalg.norm(tcp - target))
        # tcp_to_obj = float(np.linalg.norm(tcp - obj))
        obj_to_target = float(np.linalg.norm(obj - target))

        in_place_margin = np.linalg.norm(self.obj_init_pos - target)
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="gaussian",
        )

        hand_margin = float(np.linalg.norm(self.hand_init_pos - obj)) + 0.1
        hand_in_place = reward_utils.tolerance(
            tcp_to_target,
            bounds=(0, 0.25 * _TARGET_RADIUS),
            margin=hand_margin,
            sigmoid="gaussian",
        )

        reward = 3 * hand_in_place + 6 * in_place

        if obj_to_target < _TARGET_RADIUS:
            reward = 10

        return (reward, obj_to_target, hand_in_place)
