import os.path as osp
import matplotlib.pyplot as plt
import cv2
import numpy as np
from smpl_numpy import SMPL
import trimesh
from pathlib import Path
import json
from tqdm import tqdm
import itertools
from transformations import rotation_matrix

PREVIEW = False
SYNTH_BODY_DIR = "F:/Documents/These/Datasets/SynthMoCap/data/synth_body"
N_IDENTITIES = 20000
N_FRAMES_PER_IDENTITY = 5
MASKS_NAME = [
    "beard",
    "eyebrows",
    "eyelashes",
    "facewear",
    "glasses",
    "hair",
    "headwear",
]
SMPLH_MODEL = SMPL("smplh/neutral/model.npz")

BODY_CONNECTIVITY = [
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
]


def format_mask_path(identity, frame, mask_name):
    return osp.join(SYNTH_BODY_DIR, f"segm_{mask_name}_{identity:07d}_{frame:03d}.png")


def format_img_path(identity, frame):
    return osp.join(SYNTH_BODY_DIR, f"img_{identity:07d}_{frame:03d}.jpg")


def format_metadata_path(identity, frame):
    return osp.join(SYNTH_BODY_DIR, f"metadata_{identity:07d}_{frame:03d}.json")


def combine_masks(identity: int, frame: int):
    parts_mask = (
        cv2.imread(format_mask_path(identity, frame, "parts"), cv2.IMREAD_GRAYSCALE) > 0
    )

    final_mask = np.zeros_like(parts_mask, dtype=bool)
    final_mask[parts_mask] = True

    for mask_name in MASKS_NAME:
        mask_path = format_mask_path(identity, frame, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 100
        final_mask[mask] = True
    return final_mask


def compute_bbox_from_mask(mask):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return x, y, w, h


def draw_skeleton(
    img: np.ndarray,
    ldmks_2d: np.ndarray,
    ldmks_vis: np.ndarray,
    thickness: int = 1,
):
    img_size = (img.shape[1], img.shape[0])

    ldmk_connection_pairs = ldmks_2d[np.asarray(BODY_CONNECTIVITY).astype(int)].astype(
        int
    )
    for p_0, p_1 in ldmk_connection_pairs:
        cv2.line(img, tuple(p_0 + 1), tuple(p_1 + 1), (0, 0, 0), thickness, cv2.LINE_AA)
    for i, (p_0, p_1) in enumerate(ldmk_connection_pairs):
        cv2.line(
            img,
            tuple(p_0),
            tuple(p_1),
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    for i, ldmk in enumerate(ldmks_2d.astype(int)):
        if np.all(ldmk > 0) and np.all(ldmk < img_size):
            cv2.circle(img, tuple(ldmk + 1), thickness + 1, (0, 0, 0), -1, cv2.LINE_AA)

            cv2.circle(
                img,
                tuple(ldmk),
                thickness + 1,
                (0, 255, 0) if ldmks_vis[i] else (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )


for identity, frame in tqdm(
    itertools.product(range(N_IDENTITIES), range(N_FRAMES_PER_IDENTITY)),
    total=N_IDENTITIES * N_FRAMES_PER_IDENTITY,
):
    metadata_fp = format_metadata_path(identity, frame)

    if not Path(metadata_fp).exists():
        print(f"Missing metadata for identity {identity}, frame {frame}, skipping.")
        continue

    metadata = json.load(open(format_metadata_path(identity, frame)))
    body_identity = np.asarray(metadata["body_identity"])
    pose = np.asarray(metadata["pose"])
    translation = np.asarray(metadata["translation"])
    world_to_camera = np.asarray(metadata["camera"]["world_to_camera"])
    camera_to_image = np.asarray(metadata["camera"]["camera_to_image"])

    mask = combine_masks(identity, frame)
    x, y, w, h = compute_bbox_from_mask(mask)

    SMPLH_MODEL.beta = body_identity[: SMPLH_MODEL.shape_dim]
    SMPLH_MODEL.theta = pose
    SMPLH_MODEL.translation = translation

    smpl_mesh = trimesh.Trimesh(
        vertices=SMPLH_MODEL.vertices, faces=SMPLH_MODEL.triangles, face_normals=SMPLH_MODEL.normals, process=False, use_embree=True
    )

    # Ignore hand joints
    joints_3d = SMPLH_MODEL.joint_positions[:22]

    # Determine visibility of joints
    world_to_cam_gl = np.linalg.inv(world_to_camera).dot(
        rotation_matrix(np.pi, [1, 0, 0])
    )
    camera_position = world_to_cam_gl[:3, 3]

    ray_directions = joints_3d - camera_position
    ray_directions = ray_directions / np.linalg.norm(
        ray_directions, axis=1, keepdims=True
    )
    ray_origins = np.tile(camera_position, (len(joints_3d), 1))
    raycast_points, index_ray, _ = smpl_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
    )
    # Reorder raycast intersection points to match the original rays order
    raycast_points = raycast_points[np.argsort(index_ray)]
    
    joints_vis = np.zeros(len(joints_3d))

    dist_threshold = 0.3 # 30 cm

    dists = np.linalg.norm(joints_3d - raycast_points, axis=1)
    joints_vis = dists <= dist_threshold

    if PREVIEW:
        img = cv2.imread(format_img_path(identity, frame), cv2.IMREAD_COLOR)

        # Project joints in image plane and get depth
        joints_homogeneous = np.concatenate(
            [joints_3d, np.ones((joints_3d.shape[0], 1))], axis=1
        )
        joints_cam_space = (world_to_camera @ joints_homogeneous.T).T
        joints_depth = joints_cam_space[:, 2]

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

        # set mesh slightly transparent
        smpl_mesh.visual.face_colors = [200, 200, 250, 100]
        # smpl_mesh.visual.face_colors = [200, 200, 250, 255]


        draw_skeleton(img, joints_2d, joints_vis, 1)
        cv2.imshow(f"Identity {identity}, Frame {frame}", img)

        # Create 3D visualization
        rays_path = trimesh.load_path(
            np.hstack((ray_origins, joints_3d)).reshape(-1, 2, 3)
        )
        rays_path.colors = np.tile([0, 255, 0, 255], (len(rays_path.entities), 1))

        raycast_points_pcl = trimesh.points.PointCloud(raycast_points)
        raycast_points_pcl.visual.vertex_colors = [0, 0, 255, 255]

        joints_points_pcl = trimesh.points.PointCloud(joints_3d)
        joints_points_pcl.visual.vertex_colors = [[0, 255, 0, 255] if v else [255, 0, 0, 255] for v in joints_vis]

        scene = trimesh.Scene(
            [
                smpl_mesh,
                rays_path,
                raycast_points_pcl,
                joints_points_pcl,
            ]
        )
        scene.camera_transform = world_to_cam_gl
        # Set the window size to be small
        scene.show(resolution=(img.shape[1], img.shape[0]))
        cv2.destroyAllWindows()
