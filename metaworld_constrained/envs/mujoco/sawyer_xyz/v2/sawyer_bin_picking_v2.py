from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld_constrained.envs.asset_path_utils import full_v2_path_for
from metaworld_constrained.envs.mujoco.sawyer_xyz.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld_constrained.envs.mujoco.utils import reward_utils
from metaworld_constrained.types import InitConfigDict


class SawyerBinPickingEnvV2(SawyerXYZEnv):
    """SawyerBinPickingEnv.

    Motivation for V2:
        V1 was often unsolvable because the cube could be located outside of
        the starting bin. It could even be near the base of the Sawyer and out
        of reach of the gripper. V2 changes the `obj_low` and `obj_high` bounds
        to fix this.
    Changelog from V1 to V2:
        - (7/20/20) Changed object initialization space
        - (7/24/20) Added Byron's XML changes
        - (11/23/20) Updated reward function to new pick-place style
    """

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        constraint_mode: Literal["static", "relative", "absolute", "random"] = "relative",
        constraint_size: float = 0.03,
        include_const_in_obs: bool = True,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.07)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.21, 0.65, 0.02)
        obj_high = (-0.03, 0.75, 0.02)
        if constraint_mode == "relative":
            # to allow for constant relative position, constraint
            # box must be at lower end of the box at the same x-coord
            obj_list = list(obj_high)
            obj_list[0] -= 1.05*(constraint_size + 0.02)
            obj_high = tuple(obj_list)
        # Small bounds around the center of the target bin
        goal_low = np.array([0.1199, 0.699, -0.001])
        goal_high = np.array([0.1201, 0.701, +0.001])

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
            "obj_init_pos": np.array([-0.12, 0.7, 0.02]),
            "hand_init_pos": np.array((0, 0.6, 0.2)),
        }
        self.goal = np.array([0.12, 0.7, 0.02])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._target_to_obj_init: float | None = None

        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
            dtype=np.float64,
        )

        self.goal_and_obj_space = Box(
            np.hstack((goal_low[:2], obj_low[:2])),
            np.hstack((goal_high[:2], obj_high[:2])),
            dtype=np.float64,
        )

        self.goal_space = Box(goal_low, goal_high, dtype=np.float64)
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_bin_picking.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            near_object,
            grasp_success,
            obj_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(obj_to_target <= 0.05),
            "near_object": float(near_object),
            "grasp_success": float(grasp_success),
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return []

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("objGeom")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("obj").xquat

    @staticmethod
    def _squares_touch(center1, size1, center2, size2):
        half_size1 = size1
        half_size2 = size2

        # Calculate the left, right, top, and bottom edges of each square
        left1, right1 = center1[0] - half_size1, center1[0] + half_size1
        bottom1, top1 = center1[1] - half_size1, center1[1] + half_size1

        left2, right2 = center2[0] - half_size2, center2[0] + half_size2
        bottom2, top2 = center2[1] - half_size2, center2[1] + half_size2

        # Check if they touch or overlap
        horizontal_touch = not (right1 < left2 or right2 < left1)
        vertical_touch = not (top1 < bottom2 or top2 < bottom1)

        return horizontal_touch and vertical_touch
    def _calc_constraint_pos(self):
        if self._last_rand_vec.shape[0] == 9:
            pos_constraint = self._last_rand_vec[-3: ]
        elif self._constraint_mode == "relative":
            # on the right edge of the box on same y-coord as obj
            x = -0.03
            y = self.obj_init_pos[1]
            pos_constraint = np.array((x, y, .02))
        elif self._constraint_mode == "random":
            # place constraint anywhere in pickup-box, but making
            # sure it's sufficiently far away from obj
            touching = True
            pos_constraint = np.array((3, 2, 2))
            while touching:
                object_pos = self.obj_init_pos
                pos_constraint = np.random.uniform(
                    self._random_reset_space.low[:3],
                    self._random_reset_space.high[:3],
                )
                touching = self._squares_touch(object_pos[:2], 1.1*0.02, pos_constraint[:2], 1.1*self._constraint_size)
        elif self._constraint_mode == "absolute":
            pos_constraint = np.array([0, 0.7, 0.02])
        else:
            pos_constraint = self.data.body("constraint_box").xipos
        return pos_constraint

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        obj_height = self.get_body_com("obj")[2]

        self.obj_init_pos = self._get_state_rand_vec()[:2]
        self.obj_init_pos = np.concatenate([self.obj_init_pos, [obj_height]])

        self._set_obj_xyz(self.obj_init_pos)
        self._target_pos = self.get_body_com("bin_goal")
        self._target_to_obj_init = None

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[Any]
    ) -> tuple[float, bool, bool, float, float, float]:
        assert (
            self.obj_init_pos is not None and self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."

        hand = obs[:3]
        obj = obs[4:7]

        target_to_obj = float(np.linalg.norm(obj - self._target_pos))
        if self._target_to_obj_init is None:
            self._target_to_obj_init = target_to_obj

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=self._target_to_obj_init,
            sigmoid="long_tail",
        )

        threshold = 0.03
        radii = [
            np.linalg.norm(hand[:2] - self.obj_init_pos[:2]),
            np.linalg.norm(hand[:2] - self._target_pos[:2]),
        ]
        # floor is a *pair* of 3D funnels centered on (1) the object's initial
        # position and (2) the desired final position
        floor = min(
            [
                0.02 * np.log(radius - threshold) + 0.2 if radius > threshold else 0.0
                for radius in radii
            ]
        )
        # prevent the hand from running into the edge of the bins by keeping
        # it above the "floor"
        above_floor = (
            1.0
            if hand[2] >= floor
            else reward_utils.tolerance(
                max(floor - hand[2], 0.0),
                bounds=(0.0, 0.01),
                margin=0.05,
                sigmoid="long_tail",
            )
        )

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            object_reach_radius=0.01,
            xz_thresh=0.01,
            desired_gripper_effort=0.7,
            high_density=True,
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        near_object = bool(np.linalg.norm(obj - hand) < 0.04)
        pinched_without_obj = bool(obs[3] < 0.43)
        lifted = bool(obj[2] - 0.02 > self.obj_init_pos[2])
        # Increase reward when properly grabbed obj
        grasp_success = near_object and lifted and not pinched_without_obj
        if grasp_success:
            reward += 1.0 + 5.0 * reward_utils.hamacher_product(above_floor, in_place)
        # Maximize reward on success
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.0

        return (
            reward,
            near_object,
            grasp_success,
            target_to_obj,
            object_grasped,
            in_place,
        )
