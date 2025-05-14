import cv2
import numpy as np
import threading
import time
import mujoco
import glfw

from mujoco import viewer
from mujoco import MjvCamera, mjtCamera, mjtCatBit, mjtObj, mj_name2id, MjrRect
from rgbd_cam import RGBD_mujoco, run_rgbd_loop
from yolo_detector import YOLODetector
from vlm import VLMSelector

import os



# ---------------- Utils ----------------
def compute_goal_position_from_bbox(box, direction, offset=40):
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    if direction == 'front':
        return (cx, y1 - offset)
    elif direction == 'back':
        return (cx, y2 + offset)
    elif direction == 'left':
        return (x1 - offset, cy)
    elif direction == 'right':
        return (x2 + offset, cy)
    return (cx, cy)

def unproject_2d_to_camera_frame(x, y, depth, fx, fy, cx, cy):
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])

def camera_to_world(local_point, camera_pos, camera_rot):
    return camera_rot @ local_point + camera_pos

def quat_to_mat(quat):
    mat = np.zeros((3, 3))
    mujoco.mju_quat2Mat(mat.ravel(), quat)
    return mat

# ---------------- Main ----------------
def main():
    model = mujoco.MjModel.from_xml_path("../../../assets/X7S/meshes/X7S_scene.xml")
    data = mujoco.MjData(model)

    rgbd = RGBD_mujoco()
    threading.Thread(target=lambda: run_rgbd_loop(model, data, rgbd), daemon=True).start()

    viewer.launch(model, data)

    detector = YOLODetector()
    vlm = VLMSelector()

    goal_prompt = "Go in front of the table."

    while True:
        rgb = rgbd.color_image
        depth = rgbd.depth_image
        if rgb is None or depth is None:
            continue
        
        #object detection with yolo
        boxes, labels = detector.detect_objects(rgb)
        #visualize 
        annotated = detector.annotate_image(rgb, boxes, labels)
        #floor segmentation with yolo
        floor_mask = detector.segment_floor(rgb)
        floor_overlay = cv2.bitwise_and(annotated, annotated, mask=floor_mask)
        combined = cv2.addWeighted(annotated, 0.7, floor_overlay, 0.3, 0)

        if boxes: #  Call vlm when there all 1 more object 
            # vlm choice 
            choice, position, response = vlm.select_goal(annotated, labels, boxes, goal_prompt)
            print("\n--- VLM Response ---\n", response)

            if choice is not None and position is not None:
                # goal coordinate based on choosen object 
                goal_px = compute_goal_position_from_bbox(boxes[choice], position)
                x, y = goal_px
                cv2.circle(combined, goal_px, 8, (255, 0, 255), -1)
                cv2.putText(combined, f"GOAL ({position})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # Depth based unprojection → 3D world coordinate 
                depth_val = depth[y, x]
                fx = fy = rgbd.f
                cx = rgbd.cx
                cy = rgbd.cy
                local_pt = unproject_2d_to_camera_frame(x, y, depth_val, fx, fy, cx, cy)

                cam_id = rgbd.cam_id
                cam_pos = model.cam_pos[cam_id]
                cam_quat = model.cam_quat[cam_id]
                cam_rot = quat_to_mat(cam_quat)
                goal_world = camera_to_world(local_pt, cam_pos, cam_rot)

                print("GOAL WORLD POSITION:", goal_world)

        cv2.imshow("Goal Detection (YOLO + VLM)", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
