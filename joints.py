import cv2
import numpy as np
import trimesh
from smpl_numpy import SMPL, SMPLOutput
from typing import Literal

import os.path as osp
import json
import matplotlib.pyplot as plt

from constants import (
    SMPL_CONNECTIVITY,
    COCO_CONNECTIVITY,
    H36M_CONNECTIVITY,
)

try:
    import trimesh.ray.ray_pyembree
except ImportError:
    print("Failed to import embree. Make sure to install embreex.")

COCO_JOINT_REGRESSOR = np.load("J_regressor_coco.npy")
H36M_JOINT_REGRESSOR = np.load("J_regressor_h36m.npy")


def _format_img_path(dataset_dir, identity, frame):
    return osp.join(dataset_dir, f"img_{identity:07d}_{frame:03d}.jpg")


def _format_metadata_path(dataset_dir, identity, frame):
    return osp.join(dataset_dir, f"metadata_{identity:07d}_{frame:03d}.json")


def _draw_skeleton_2d(img, joints_2d, joints_vis, connectivity, thickness=1):
    img_size = (img.shape[1], img.shape[0])

    connection_pairs = joints_2d[connectivity].astype(int)
    for p_0, p_1 in connection_pairs:
        cv2.line(img, tuple(p_0 + 1), tuple(p_1 + 1), (0, 0, 0), thickness, cv2.LINE_AA)
    for i, (p_0, p_1) in enumerate(connection_pairs):
        cv2.line(
            img,
            tuple(p_0),
            tuple(p_1),
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    for i, joint_2d in enumerate(joints_2d.astype(int)):
        if np.all(joint_2d > 0) and np.all(joint_2d < img_size):

            cv2.putText(
                img,
                str(i),
                (joint_2d[0] + 2, joint_2d[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.circle(
                img, tuple(joint_2d + 1), thickness + 1, (0, 0, 0), -1, cv2.LINE_AA
            )

            cv2.circle(
                img,
                tuple(joint_2d),
                thickness + 1,
                (0, 255, 0) if joints_vis[i] else (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )


def _project_joints(joints_3d, world_to_camera, camera_to_image):
    joints_homogeneous = np.concatenate(
        [joints_3d, np.ones((joints_3d.shape[0], 1))], axis=1
    )
    joints_cam_space = (world_to_camera @ joints_homogeneous.T).T

    joints_2d_homogeneous = (
        joints_cam_space[:, :3] / joints_cam_space[:, 2][:, np.newaxis]
    )

    K = np.array(
        [
            [camera_to_image[0, 0], 0, camera_to_image[0, 2]],
            [0, camera_to_image[1, 1], camera_to_image[1, 2]],
            [0, 0, 1],
        ]
    )
    joints_2d = (K @ joints_2d_homogeneous.T).T
    joints_2d = joints_2d[:, :2]
    return joints_2d


def _compute_joints_visibility(
    smplh_output: SMPLOutput,
    world_to_camera: np.ndarray,
    joints_3d: np.ndarray,
    joints_2d: np.ndarray,
    visibility_threshold: float,
    mask: np.ndarray,
    preview_3d: bool = False,
    preview_resolution: np.ndarray = np.array([512, 512]),
) -> np.ndarray:
    # Convert camera to OpenGL convention: pi rotation around x axis
    R = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )
    world_to_cam_gl = np.linalg.inv(world_to_camera).dot(R)
    camera_position = world_to_cam_gl[:3, 3]

    smpl_mesh = trimesh.Trimesh(
        vertices=smplh_output.vertices,
        faces=smplh_output.faces,
        face_normals=smplh_output.normals,
        process=False,
        use_embree=True,
    )

    ray_directions = joints_3d - camera_position
    ray_directions = ray_directions / np.linalg.norm(
        ray_directions, axis=1, keepdims=True
    )
    ray_origins = np.tile(camera_position, (len(joints_3d), 1))
    ray_intersections, index_ray, _ = smpl_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
    )

    raycast_points = np.zeros_like(joints_3d)
    raycast_points[index_ray] = ray_intersections
    dists = np.linalg.norm(joints_3d - raycast_points, axis=1)

    self_occlusion = dists > visibility_threshold

    # Externally occluded if joints_2d is not in the mask or outside the image
    external_occlusion = np.zeros(len(joints_2d), dtype=bool)
    for i, joint_2d in enumerate(joints_2d):
        x, y = joint_2d.astype(int)
        if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0] or not mask[y, x]:
            external_occlusion[i] = True

    joints_vis = np.logical_not(self_occlusion | external_occlusion)

    if preview_3d:
        smpl_mesh.visual.face_colors = [200, 200, 250, 100]

        rays = trimesh.load_path(np.hstack((ray_origins, joints_3d)).reshape(-1, 2, 3))
        rays.colors = np.tile([0, 255, 0, 255], (len(rays.entities), 1))
        rays_intersections_pcl = trimesh.points.PointCloud(raycast_points)
        rays_intersections_pcl.visual.vertex_colors = [0, 0, 255, 255]

        joints_pcl = trimesh.points.PointCloud(joints_3d)
        joints_pcl.visual.vertex_colors = [
            [255, 0, 0, 255] if v else [0, 255, 0, 255] for v in self_occlusion
        ]

        scene = trimesh.Scene(
            [
                smpl_mesh,
                rays,
                rays_intersections_pcl,
                joints_pcl,
            ]
        )
        scene.camera_transform = world_to_cam_gl
        scene.show(resolution=preview_resolution, line_settings={"point_size": 5})

    return joints_vis


def generate_joints_annotations(
    smplh_model: SMPL,
    skeleton_type: Literal["smpl", "coco", "h36m"],
    identity: int,
    frame: int,
    segmentation_mask: np.ndarray,
    body_identity: np.ndarray,
    body_pose: np.ndarray,
    body_translation: np.ndarray,
    camera_w2c: np.ndarray,
    camera_K: np.ndarray,
    camera_resolution: np.ndarray,
    visibility_threshold: float,
    preview_2d_img: np.ndarray = None,
    preview_2d: bool = False,
    preview_3d: bool = False,
):
    if skeleton_type == "smpl":
        regressor = smplh_model.joint_regressor
        connectivity = SMPL_CONNECTIVITY
    elif skeleton_type == "coco":
        regressor = COCO_JOINT_REGRESSOR
        connectivity = COCO_CONNECTIVITY
    elif skeleton_type == "h36m":
        regressor = H36M_JOINT_REGRESSOR
        connectivity = H36M_CONNECTIVITY
    else:
        raise ValueError("Invalid skeleton type.")

    betas = body_identity[: smplh_model.shape_dim]
    smplh_output = smplh_model.forward(betas, body_pose, body_translation)

    joints_3d = smplh_output.compute_joints_positions(regressor)
    joints_2d = _project_joints(joints_3d, camera_w2c, camera_K)

    joints_vis = _compute_joints_visibility(
        smplh_output=smplh_output,
        world_to_camera=camera_w2c,
        joints_3d=joints_3d,
        joints_2d=joints_2d,
        visibility_threshold=visibility_threshold,
        mask=segmentation_mask,
        preview_3d=preview_3d,
        preview_resolution=camera_resolution,
    )

    if preview_2d:
        if preview_2d_img is None:
            raise ValueError("Preview 2D image is required for 2D preview.")

        img = preview_2d_img.copy()
        _draw_skeleton_2d(img, joints_2d, joints_vis, connectivity)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"SMPL Skeleton - Identity {identity}, frame {frame}")
        plt.axis("off")
        plt.show()

    return joints_3d, joints_2d, joints_vis
