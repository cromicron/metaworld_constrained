from __future__ import annotations

from typing import Any, Literal

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld_constrained.envs.asset_path_utils import full_v2_path_for
from metaworld_constrained.envs.mujoco.sawyer_xyz.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld_constrained.envs.mujoco.utils import reward_utils
from metaworld_constrained.types import InitConfigDict


class SawyerDialTurnEnvV2(SawyerXYZEnv):
    TARGET_RADIUS: float = 0.07

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
        obj_low = (-0.1, 0.80, 0.0)
        obj_high = (0.1, 0.90, 0.0)
        goal_low = (-0.1, 0.83, 0.0299)
        goal_high = (0.1, 0.93, 0.0301)

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
            "obj_init_pos": np.array([0, 0.7, 0.0]),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.73, 0.08])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_dial.xml")

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
            "success": float(target_to_obj <= self.TARGET_RADIUS),
            "near_object": float(tcp_to_obj <= 0.01),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        dial_center = self.get_body_com("dial").copy()
        dial_angle_rad = self.data.joint("knob_Joint_1").qpos

        offset = np.array(
            [np.sin(dial_angle_rad).item(), -np.cos(dial_angle_rad).item(), 0.0]
        )
        dial_radius = 0.05

        offset *= dial_radius

        return dial_center + offset

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("dial").xquat

    import numpy as np

    def _calc_constraint_pos(self):
        if self._constraint_mode == "random":
            # Circle parameters
            center = self.obj_init_pos[:2]  # Circle center coordinates as a NumPy array
            r = 0.05  # Circle radius (dial_radius)

            # Square parameters
            h = self._constraint_size / 2  # Half-length of the square's edge

            # Distance constraints
            D_min = self._constraint_size  # Minimum distance from the circle to the square
            D_max = 0.1  # Maximum distance from the circle to the square

            # Place the square randomly using vectorized operations
            square_pos = self._place_square_randomly_vectorized(center, r, h, D_min, D_max)

            # Return the position with a fixed z-coordinate (assuming z = 0.1)
            return np.append(square_pos, 0.1)
        else:
            return super()._calc_constraint_pos()

    def _place_square_randomly_vectorized(self, center, r, h, D_min, D_max):
        """
        Randomly place the square using vectorized operations such that it satisfies the distance constraints.
        """
        # Define the search area
        max_extent = D_max + r + h * np.sqrt(2)
        min_sample = center - max_extent
        max_sample = center + max_extent

        # Number of samples to generate in one batch
        num_samples = 10000

        # Generate random samples within the search area
        samples = np.random.uniform(min_sample, max_sample, size=(num_samples, 2))

        # Apply the y constraint (vectorized)
        y_constraint = samples[:, 1] <= self.obj_init_pos[1] + 0.05
        samples = samples[y_constraint]

        # If no samples are valid after applying y constraint, raise an error
        if len(samples) == 0:
            raise ValueError("No valid samples after applying y constraint.")

        # Compute the square boundaries for all samples (vectorized)
        square_min = samples - h  # Lower-left corner of the squares
        square_max = samples + h  # Upper-right corner of the squares

        # Compute dx and dy for all samples (vectorized)
        # For dx
        dx = np.maximum.reduce([
            square_min[:, 0] - center[0],  # When circle center is left of square
            np.zeros(len(samples)),  # When circle center is within square horizontally
            center[0] - square_max[:, 0]  # When circle center is right of square
        ])

        # For dy
        dy = np.maximum.reduce([
            square_min[:, 1] - center[1],  # When circle center is below square
            np.zeros(len(samples)),  # When circle center is within square vertically
            center[1] - square_max[:, 1]  # When circle center is above square
        ])

        # Compute the minimal distances D for all samples (vectorized)
        d_center_to_square = np.sqrt(dx ** 2 + dy ** 2)
        D = d_center_to_square - r

        # Find samples that satisfy the distance constraints
        valid_indices = (D >= D_min) & (D <= D_max)

        # If valid samples are found, select one at random
        if np.any(valid_indices):
            idx = np.random.choice(np.where(valid_indices)[0])
            square_pos = samples[idx]
            return square_pos
        else:
            # If no valid position is found, raise an error
            raise ValueError("Could not find a valid square position within the generated samples.")

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        goal_pos = self._get_state_rand_vec()[:3]
        self.obj_init_pos = goal_pos[:3]
        final_pos = goal_pos.copy() + np.array([0, 0.03, 0.03])
        self._target_pos = final_pos
        self.model.body("dial").pos = self.obj_init_pos
        self.dial_push_position = self._get_pos_objects() + np.array([0.05, 0.02, 0.09])
        self.model.site("goal").pos = self._target_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        obj = self._get_pos_objects()
        dial_push_position = self._get_pos_objects() + np.array([0.05, 0.02, 0.09])
        tcp = self.tcp_center
        target = self._target_pos.copy()

        target_to_obj = obj - target
        target_to_obj = float(np.linalg.norm(target_to_obj).item())
        target_to_obj_init = self.dial_push_position - target
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )

        dial_reach_radius = 0.005
        tcp_to_obj = float(np.linalg.norm(dial_push_position - tcp).item())
        tcp_to_obj_init = float(
            np.linalg.norm(self.dial_push_position - self.init_tcp).item()
        )
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, dial_reach_radius),
            margin=abs(tcp_to_obj_init - dial_reach_radius),
            sigmoid="gaussian",
        )
        gripper_closed = min(max(0, action[-1]), 1)

        reach = reward_utils.hamacher_product(reach, gripper_closed)
        tcp_opened = 0
        object_grasped = reach

        reward = 10 * reward_utils.hamacher_product(reach, in_place)
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place,
        )
