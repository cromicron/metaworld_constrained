from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld_constrained.envs.asset_path_utils import full_v2_path_for
from metaworld_constrained.envs.mujoco.sawyer_xyz.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld_constrained.envs.mujoco.utils import reward_utils
from metaworld_constrained.types import InitConfigDict


class SawyerDrawerCloseEnvV2(SawyerXYZEnv):
    _TARGET_RADIUS: float = 0.04

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        constraint_mode: Literal["static", "relative", "absolute", "random"] = "relative",
        constraint_size: float = 0.03,
        include_const_in_obs: bool = True,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.0)
        obj_high = (0.1, 0.9, 0.0)

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
            "obj_init_pos": np.array([0.0, 0.9, 0.0], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

        self.maxDist = 0.15
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_drawer.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            _,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(target_to_obj <= self.TARGET_RADIUS + 0.015),
            "near_object": float(tcp_to_obj <= 0.01),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.05])

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.zeros(4)

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
            # constraint is underneath the handle
            handle_pos = self._get_pos_objects()[: -1] + np.array([0, 0.03])
            x_middle = (self.hand_init_pos[0] + handle_pos[0]) / 2
            pos_constraint = np.hstack((x_middle, handle_pos[1], [0.02]))
        elif self._constraint_mode == "random":
            # place constraint in box demarcated by drawer on x coordinate
            # and the distance of hand and drawer edge on y

            goal_pos = self._get_pos_objects()[:-1]
            x_min = goal_pos[0] -0.12
            x_max = goal_pos[0] +0.12
            y_min = self.hand_init_pos[1]
            y_max = goal_pos[1] + 0.03

            pos_constraint = np.random.uniform(
                np.array([x_min, y_min, 0.02]),
                np.array([x_max, y_max, 0.02]),
                size=3,
            )
        elif self._constraint_mode == "absolute":
            pos_constraint = np.array([0, 0.7, 0.02])
        else:
            pos_constraint = self.data.body("constraint_box").xipos
        return pos_constraint

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        # Compute nightstand position
        self.obj_init_pos = self._get_state_rand_vec()[: 3]
        # Set mujoco body to computed position

        self.model.body("drawer").pos = self.obj_init_pos
        # Set _target_pos to current drawer position (closed)
        self._target_pos = self.obj_init_pos + np.array([0.0, -0.16, 0.09])
        # Pull drawer out all the way and mark its starting position
        self._set_obj_xyz(np.array(-self.maxDist))
        self.obj_init_pos = self._get_pos_objects()
        self.model.site("goal").pos = self._target_pos
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None and self.hand_init_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        obj = obs[4:7]

        tcp = self.tcp_center
        target = self._target_pos.copy()

        target_to_obj = obj - target
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = self.obj_init_pos - target
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )

        handle_reach_radius = 0.005
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_reach_radius),
            margin=abs(tcp_to_obj_init - handle_reach_radius),
            sigmoid="gaussian",
        )
        gripper_closed = min(max(0, action[-1]), 1)

        reach = reward_utils.hamacher_product(reach, gripper_closed)
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        if target_to_obj <= self.TARGET_RADIUS + 0.015:
            reward = 1.0

        reward *= 10

        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)
