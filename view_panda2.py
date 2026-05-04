"""Open panda_demo_2.xml in MuJoCo viewer, frozen at the home keyframe."""

import os
import time
import mujoco
import mujoco.viewer

XML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "franka_panda", "franka_emika_panda", "scene_demo_2.xml",
)

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 150
    viewer.cam.elevation = -20
    viewer.cam.distance = 1.4
    viewer.cam.lookat[:] = [0.4, 0.0, 0.3]
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.016)   # ~60 Hz refresh, no physics step
