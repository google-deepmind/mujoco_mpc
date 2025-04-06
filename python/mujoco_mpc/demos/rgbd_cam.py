import mujoco
from mujoco import viewer
import numpy as np
import cv2
import threading
import time
import glfw

from mujoco import MjvCamera, mjtCamera, mjtCatBit, mj_name2id, mjtObj, MjrRect

class RGBD_mujoco:
    def __init__(self):
        self.cx = self.cy = self.f = 0
        self.z_near = self.z_far = self.extent = 1
        self.color_image = None
        self.depth_image = None

    def set_intrinsics(self, model, cam, viewport):
        fovy = model.cam_fovy[cam.fixedcamid] / 180 * np.pi / 2
        self.f = viewport.height / 2 / np.tan(fovy)
        self.cx = viewport.width / 2
        self.cy = viewport.height / 2

    def linearize_depth(self, raw):
        depth = np.zeros_like(raw, dtype=np.float32)
        mask = raw > 0
        depth[mask] = self.z_near * self.z_far * self.extent / (self.z_far - raw[mask] * (self.z_far - self.z_near))
        return depth

    def get_rgbd(self, model, data, cam, context, viewport, scene):
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), mujoco.MjvPerturb(), cam, mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        rgb = np.empty((viewport.height, viewport.width, 3), dtype=np.uint8)
        depth = np.empty((viewport.height, viewport.width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, viewport, context)

        self.z_near = model.vis.map.znear
        self.z_far = model.vis.map.zfar
        self.extent = model.stat.extent

        self.color_image = cv2.flip(rgb, 0)
        raw_depth = cv2.flip(depth, 0)
        self.depth_image = self.linearize_depth(raw_depth)

def run_rgbd_loop(model, data):
    glfw.init()
    window = glfw.create_window(640, 480, "MuJoCo RGBD", None, None)
    glfw.make_context_current(window)

    cam = MjvCamera()
    cam.type = mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mj_name2id(model, mjtObj.mjOBJ_CAMERA, "head_cam")

    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    viewport = MjrRect(0, 0, 640, 480)

    scene = mujoco.MjvScene(model, maxgeom=1000)

    rgbd = RGBD_mujoco()
    rgbd.set_intrinsics(model, cam, viewport)

    while True:
        #mujoco.mj_step(model, data)  # Simulation step

        rgbd.get_rgbd(model, data, cam, context, viewport, scene)
        img = rgbd.color_image

        if img is not None:
            cv2.imshow("head_cam RGB", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(1.0 / 30)

def main():
    model = mujoco.MjModel.from_xml_path("../../../assets/X7S/meshes/X7S_scene.xml") 
    data = mujoco.MjData(model)


    threading.Thread(target=lambda: run_rgbd_loop(model, data), daemon=True).start()

    viewer.launch(model, data)

if __name__ == "__main__":
    main()