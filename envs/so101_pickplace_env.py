"""
Gymnasium environment wrapping the SO-101 MuJoCo pick-and-place scene.

Task: pick the small object off the tray and place it near a target position.
Observation: joint positions + velocities + object 3D position (state-only, no camera yet).
Action: position targets for the 6 SO-101 joints.
"""

import os
import time
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

# Joint / actuator names, in the order defined in so101_new_calib.xml
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Matches <position ... ctrlrange="..."/> in so101_new_calib.xml
ACTION_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453], dtype=np.float32)
ACTION_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533], dtype=np.float32)

# Where the object should end up to count as "placed" — center of the place_bin body
PLACE_TARGET = np.array([-0.14, 0.14, 0.02], dtype=np.float32)
# Bin interior is ~7cm x 7cm (half-size 0.035); tolerance covers landing anywhere inside it
PLACE_TOLERANCE = 0.03  # ~3 cm, keeps success inside the bin walls

# Reset randomization range for the object's starting (x, y) on the tray
OBJECT_RESET_XY_RANGE = 0.03  # +/- 3 cm around the nominal (0.2, 0.0)


class SO101PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, scene_path: str, render_mode: str | None = None, max_steps: int = 300):
        super().__init__()

        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene file not found: {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.max_steps = max_steps
        self._step_count = 0
        self._viewer = None
        self._last_render_time = 0.0

        # Resolve joint/actuator IDs once, fail loudly if names don't match the XML
        self._joint_qpos_addr = []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found in model. Check XML joint names.")
            self._joint_qpos_addr.append(self.model.jnt_qposadr[jid])

        self._actuator_ids = []
        for name in JOINT_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                raise ValueError(f"Actuator '{name}' not found in model. Check XML actuator names.")
            self._actuator_ids.append(aid)

        self._object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        if self._object_body_id == -1:
            raise ValueError("Body 'target_object' not found in model. Check scene XML.")
        self._object_qpos_addr = self.model.jnt_qposadr[
            self.model.body_jntadr[self._object_body_id]
        ]

        # Action space: 6 joint position targets
        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)

        # Observation: 6 joint pos + 6 joint vel + 3 object pos = 15 values
        obs_dim = len(JOINT_NAMES) * 2 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        joint_pos = np.array(
            [self.data.qpos[addr] for addr in self._joint_qpos_addr], dtype=np.float32
        )
        joint_vel = np.array(
            [self.data.qvel[addr] for addr in self._joint_qpos_addr], dtype=np.float32
        )
        object_pos = self.data.qpos[self._object_qpos_addr : self._object_qpos_addr + 3].astype(
            np.float32
        )
        return np.concatenate([joint_pos, joint_vel, object_pos])

    def _get_object_pos(self) -> np.ndarray:
        return self.data.qpos[self._object_qpos_addr : self._object_qpos_addr + 3].astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        def _do_reset():
            mujoco.mj_resetData(self.model, self.data)

            # Randomize object starting position slightly, on top of the tray
            rng = self.np_random
            x_offset = rng.uniform(-OBJECT_RESET_XY_RANGE, OBJECT_RESET_XY_RANGE)
            y_offset = rng.uniform(-OBJECT_RESET_XY_RANGE, OBJECT_RESET_XY_RANGE)
            self.data.qpos[self._object_qpos_addr] = 0.20 + x_offset
            self.data.qpos[self._object_qpos_addr + 1] = 0.00 + y_offset
            self.data.qpos[self._object_qpos_addr + 2] = 0.02
            # Freejoint quaternion (w,x,y,z) — identity orientation
            self.data.qpos[self._object_qpos_addr + 3 : self._object_qpos_addr + 7] = [1, 0, 0, 0]

            mujoco.mj_forward(self.model, self.data)

        if self._viewer is not None:
            with self._viewer.lock():
                _do_reset()
        else:
            _do_reset()

        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, ACTION_LOW, ACTION_HIGH)

        if self._viewer is not None:
            # Passive viewer renders on a background thread — writes to data
            # and mj_step must be locked, or MuJoCo segfaults on a data race.
            with self._viewer.lock():
                for aid, target in zip(self._actuator_ids, action):
                    self.data.ctrl[aid] = target
                mujoco.mj_step(self.model, self.data)
        else:
            for aid, target in zip(self._actuator_ids, action):
                self.data.ctrl[aid] = target
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        object_pos = self._get_object_pos()

        dist_to_target = np.linalg.norm(object_pos - PLACE_TARGET)
        success = bool(dist_to_target < PLACE_TOLERANCE)
        # Sparse reward: 1.0 on success, small negative distance shaping otherwise
        reward = 1.0 if success else -float(dist_to_target)

        terminated = success
        truncated = self._step_count >= self.max_steps

        info = {"success": success, "distance_to_target": float(dist_to_target)}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._last_render_time = time.time()
            return
        # Decouple render rate from physics step rate. Physics may step at
        # 500Hz+; syncing the viewer that often overwhelms the render thread
        # and was the actual cause of the earlier segfaults. Only sync at
        # roughly render_fps, skipping most physics steps' frames.
        now = time.time()
        render_interval = 1.0 / self.metadata["render_fps"]
        if now - self._last_render_time < render_interval:
            return
        self._viewer.sync()
        self._last_render_time = now

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None