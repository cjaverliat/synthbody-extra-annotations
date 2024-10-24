import cv2
import numpy as np
import trimesh
from smpl_numpy import SMPL, SMPLOutput

import os.path as osp
import json
import matplotlib.pyplot as plt

try:
    import trimesh.ray.ray_pyembree
except ImportError:
    print("Failed to import embree. Make sure to install embreex.")

COCO_JOINT_REGRESSOR = np.load("J_regressor_coco.npy")
H36M_JOINT_REGRESSOR = np.load("J_regressor_h36m.npy")

SMPL_CONNECTIVITY = [
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 1],
    [5, 2],
    [6, 3],
    [7, 4],
    [8, 5],
    [9, 6],
    [10, 7],
    [11, 8],
    [12, 9],
    [13, 9],
    [14, 9],
    [15, 12],
    [16, 13],
    [17, 14],
    [18, 16],
    [19, 17],
    [20, 18],
    [21, 19],
    [22, 20],
    [23, 22],
    [24, 23],
    [25, 20],
    [26, 25],
    [27, 26],
    [28, 20],
    [29, 28],
    [30, 29],
    [31, 20],
    [32, 31],
    [33, 32],
    [34, 20],
    [35, 34],
    [36, 35],
    [37, 21],
    [38, 37],
    [39, 38],
    [40, 21],
    [41, 40],
    [42, 41],
    [43, 21],
    [44, 43],
    [45, 44],
    [46, 21],
    [47, 46],
    [48, 47],
    [49, 21],
    [50, 49],
    [51, 50],
]

COCO_CONNECTIVITY = [
    [0, 1],
    [0, 2],
    [2, 1],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [6, 12],
    [5, 11],
    [12, 11],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
]

H36M_CONNECTIVITY = [
    [0, 1],
    [0, 4],
    [4, 5],
    [5, 6],
    [1, 2],
    [2, 3],
    [0, 7],
    [7, 8],
    [8, 11],
    [11, 12],
    [12, 13],
    [8, 14],
    [14, 15],
    [15, 16],
    [8, 9],
    [9, 10],
]


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
    preview_resolution: tuple[int, int] = (512, 512),
):
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

    # 0 if visible, 1 if self-occluded, 2 if externally occluded, 3 if both
    joints_vis_flags = np.zeros(len(joints_vis), dtype=np.uint8)
    joints_vis_flags[self_occlusion] = 1
    joints_vis_flags[external_occlusion] = 2
    joints_vis_flags[self_occlusion & external_occlusion] = 3

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

    return joints_vis, joints_vis_flags


def generate_joints_annotations(
    smplh_model: SMPL,
    dataset_dir: str,
    identity: int,
    frame: int,
    visibility_threshold: float,
    mask: np.ndarray,
    preview_2d: bool,
    preview_3d: bool,
):
    metadata_fp = _format_metadata_path(dataset_dir, identity, frame)
    metadata = json.load(open(metadata_fp))

    body_identity = np.asarray(metadata["body_identity"])
    pose = np.asarray(metadata["pose"])
    translation = np.asarray(metadata["translation"])
    world_to_camera = np.asarray(metadata["camera"]["world_to_camera"])
    camera_to_image = np.asarray(metadata["camera"]["camera_to_image"])
    resolution = np.asarray(metadata["camera"]["resolution"])

    betas = body_identity[: smplh_model.shape_dim]
    smplh_output = smplh_model.forward(betas, pose, translation)

    smpl_joints_3d = smplh_output.compute_joints_positions(smplh_model.joint_regressor)
    smpl_joints_2d = _project_joints(smpl_joints_3d, world_to_camera, camera_to_image)

    coco_joints_3d = smplh_output.compute_joints_positions(COCO_JOINT_REGRESSOR)
    coco_joints_2d = _project_joints(coco_joints_3d, world_to_camera, camera_to_image)

    h36m_joints_3d = smplh_output.compute_joints_positions(H36M_JOINT_REGRESSOR)
    h36m_joints_2d = _project_joints(h36m_joints_3d, world_to_camera, camera_to_image)

    smpl_joints_vis, smpl_joints_vis_flags = _compute_joints_visibility(
        smplh_output=smplh_output,
        world_to_camera=world_to_camera,
        joints_3d=smpl_joints_3d,
        joints_2d=smpl_joints_2d,
        visibility_threshold=visibility_threshold,
        mask=mask,
        preview_3d=preview_3d,
        preview_resolution=resolution,
    )

    coco_joints_vis, coco_joints_vis_flags = _compute_joints_visibility(
        smplh_output=smplh_output,
        world_to_camera=world_to_camera,
        joints_3d=coco_joints_3d,
        joints_2d=coco_joints_2d,
        visibility_threshold=visibility_threshold,
        mask=mask,
        preview_3d=preview_3d,
        preview_resolution=resolution,
    )

    h36m_joints_vis, h36m_joints_vis_flags = _compute_joints_visibility(
        smplh_output=smplh_output,
        world_to_camera=world_to_camera,
        joints_3d=h36m_joints_3d,
        joints_2d=h36m_joints_2d,
        visibility_threshold=visibility_threshold,
        mask=mask,
        preview_3d=preview_3d,
        preview_resolution=resolution,
    )

    if preview_2d:
        img = cv2.imread(_format_img_path(dataset_dir, identity, frame))

        smpl_img = img.copy()
        coco_img = img.copy()
        h36m_img = img.copy()
        _draw_skeleton_2d(smpl_img, smpl_joints_2d, smpl_joints_vis, SMPL_CONNECTIVITY)
        _draw_skeleton_2d(coco_img, coco_joints_2d, coco_joints_vis, COCO_CONNECTIVITY)
        _draw_skeleton_2d(h36m_img, h36m_joints_2d, h36m_joints_vis, H36M_CONNECTIVITY)

        plt.subplot(1, 3, 1)
        plt.title(f"SMPL Skeleton - Identity {identity}, frame {frame}")
        plt.axis("off")
        plt.imshow(cv2.cvtColor(smpl_img, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 2)
        plt.title(f"COCO Skeleton - Identity {identity}, frame {frame}")
        plt.axis("off")
        plt.imshow(cv2.cvtColor(coco_img, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 3)
        plt.title(f"H36M Skeleton - Identity {identity}, frame {frame}")
        plt.axis("off")
        plt.imshow(cv2.cvtColor(h36m_img, cv2.COLOR_BGR2RGB))

        plt.show()

    return {
        "smpl": {
            "joints_3d": smpl_joints_3d.tolist(),
            "joints_2d": smpl_joints_2d.tolist(),
            "joints_vis": smpl_joints_vis.tolist(),
            "joints_vis_flags": smpl_joints_vis_flags.tolist(),
        },
        "coco": {
            "joints_3d": coco_joints_3d.tolist(),
            "joints_2d": coco_joints_2d.tolist(),
            "joints_vis": coco_joints_vis.tolist(),
            "joints_vis_flags": coco_joints_vis_flags.tolist(),
        },
        "h36m": {
            "joints_3d": h36m_joints_3d.tolist(),
            "joints_2d": h36m_joints_2d.tolist(),
            "joints_vis": h36m_joints_vis.tolist(),
            "joints_vis_flags": h36m_joints_vis_flags.tolist(),
        },
    }
