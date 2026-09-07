import os
import mujoco
import mujoco.viewer

scene_path = os.path.join(os.path.dirname(__file__), '..', 'scenes', 'watch_scene.xml')
scene_path = os.path.abspath(scene_path)

model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
