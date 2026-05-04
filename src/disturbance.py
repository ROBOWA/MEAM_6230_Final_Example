"""External force disturbance applied to a robot body COM during a time window.

Usage in a simulation loop (before mj_step):
    disturbance = Disturbance(env.model, body_name, start_time, duration, force)
    active = disturbance.apply(env.data, sim_time)
"""

import numpy as np
import mujoco


class Disturbance:
    """Applies a world-frame force to a MuJoCo body COM for a fixed time window.

    The force acts at the body COM and is expressed in the world frame.
    Outside the window data.xfrc_applied[body_id] is cleared to zero.
    """

    def __init__(self, model, body_name, start_time, duration, force, enabled=True):
        """
        Args:
            model:      mujoco.MjModel
            body_name:  name of the body to push (e.g. "link5")
            start_time: simulation time [s] when force switches on
            duration:   how long [s] the force lasts
            force:      world-frame force vector (3,) [N]
            enabled:    master switch; if False, nothing is ever applied
        """
        self.enabled    = bool(enabled)
        self.body_name  = body_name
        self.start_time = float(start_time)
        self.duration   = float(duration)
        self.force      = np.asarray(force, dtype=float)

        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if self.body_id < 0:
            raise ValueError(f"Disturbance: body '{body_name}' not found in model.")

    @property
    def end_time(self):
        return self.start_time + self.duration

    def apply(self, data, sim_time):
        """Apply or clear the disturbance force.

        Args:
            data:     mujoco.MjData (modified in-place)
            sim_time: current simulation time [s]

        Returns:
            True if the force is currently active.
        """
        active = (
            self.enabled
            and self.start_time <= sim_time < self.end_time
        )
        if active:
            data.xfrc_applied[self.body_id, :3] = self.force
            data.xfrc_applied[self.body_id, 3:] = 0.0
        else:
            data.xfrc_applied[self.body_id, :] = 0.0
        return active


def apply_external_disturbance(disturbance, env, sim_time):
    """Convenience wrapper: apply disturbance and return True if force is active.

    Args:
        disturbance: Disturbance instance
        env:         simulation environment (must expose .data: mujoco.MjData)
        sim_time:    current simulation time [s]
    """
    return disturbance.apply(env.data, sim_time)
