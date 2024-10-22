import os.path as osp
import cv2
import numpy as np
from smpl_numpy import SMPL
import trimesh
from pathlib import Path
import json
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import trimesh.ray.ray_pyembree
except ImportError:
    print("Failed to import embree. Make sure to install embreex.")

VISIBILITY_THRESHOLD = 0.3  # 30 cm
PREVIEW = True
SYNTH_BODY_DIR = "F:/Documents/These/Datasets/SynthMoCap/data/synth_body"
OUTPUT_DIR = "F:/Documents/These/Datasets/SynthMoCap/data/synth_body_extras/"
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

coco_joint_regressor = np.load("J_regressor_coco.npy")
h36m_joint_regressor = np.load("J_regressor_h36m.npy")

SMPLH_MODEL = SMPL("F:/Documents/These/Datasets/SMPL/models/smplh/SMPLH_NEUTRAL.npz")

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
    [9, 10]
]

def format_output_mask_path(identity, frame):
    return osp.join(OUTPUT_DIR, f"mask_{identity:07d}_{frame:03d}.png")


def format_output_extra_metadata_path(identity, frame):
    return osp.join(OUTPUT_DIR, f"extra_metadata_{identity:07d}_{frame:03d}.json")


def format_segm_mask_path(identity, frame, mask_name):
    return osp.join(SYNTH_BODY_DIR, f"segm_{mask_name}_{identity:07d}_{frame:03d}.png")


def format_img_path(identity, frame):
    return osp.join(SYNTH_BODY_DIR, f"img_{identity:07d}_{frame:03d}.jpg")


def format_metadata_path(identity, frame):
    return osp.join(SYNTH_BODY_DIR, f"metadata_{identity:07d}_{frame:03d}.json")


def combine_masks(identity: int, frame: int):
    parts_mask = (
        cv2.imread(
            format_segm_mask_path(identity, frame, "parts"), cv2.IMREAD_GRAYSCALE
        )
        > 0
    )

    final_mask = np.zeros_like(parts_mask, dtype=bool)
    final_mask[parts_mask] = True

    for mask_name in MASKS_NAME:
        mask_path = format_segm_mask_path(identity, frame, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 100
        final_mask[mask] = True
    return final_mask


def compute_bbox_from_mask(mask):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return x, y, w, h


def project_joints(joints_3d, world_to_camera, camera_to_image):
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


def compute_joints_visibility(
    smplh_model: SMPL,
    world_to_camera: np.ndarray,
    joints_3d: np.ndarray,
    joints_2d: np.ndarray,
    visibility_threshold: float,
    mask: np.ndarray,
    show_preview: bool = False,
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
        vertices=smplh_model.vertices,
        faces=smplh_model.triangles,
        face_normals=smplh_model.normals,
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

    if show_preview:
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


def draw_skeleton_2d(img, joints_2d, joints_vis, connectivity, thickness=1):
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


def process_identity_frame(smplh_model: SMPL, identity, frame):
    metadata_fp = format_metadata_path(identity, frame)

    if not Path(metadata_fp).exists():
        print(f"Missing metadata for identity {identity}, frame {frame}, skipping.")
        return

    metadata = json.load(open(format_metadata_path(identity, frame)))
    body_identity = np.asarray(metadata["body_identity"])
    pose = np.asarray(metadata["pose"])
    translation = np.asarray(metadata["translation"])
    world_to_camera = np.asarray(metadata["camera"]["world_to_camera"])
    camera_to_image = np.asarray(metadata["camera"]["camera_to_image"])
    resolution = np.asarray(metadata["camera"]["resolution"])

    mask = combine_masks(identity, frame)
    bbox = compute_bbox_from_mask(mask)  # Bbox in format xywh

    smplh_model.beta = body_identity[: smplh_model.shape_dim]
    smplh_model.theta = pose
    smplh_model.translation = translation

    vertices = smplh_model.vertices

    smpl_joints_3d = smplh_model.joint_positions
    smpl_joints_2d = project_joints(smpl_joints_3d, world_to_camera, camera_to_image)

    coco_joints_3d = coco_joint_regressor.dot(vertices)
    coco_joints_2d = project_joints(coco_joints_3d, world_to_camera, camera_to_image)

    h36m_joints_3d = h36m_joint_regressor.dot(vertices)
    h36m_joints_2d = project_joints(h36m_joints_3d, world_to_camera, camera_to_image)

    smpl_joints_vis, smpl_joints_vis_flags = compute_joints_visibility(
        smplh_model,
        world_to_camera,
        smpl_joints_3d,
        smpl_joints_2d,
        VISIBILITY_THRESHOLD,
        mask,
        show_preview=PREVIEW,
        preview_resolution=resolution,
    )

    coco_joints_vis, coco_joints_vis_flags = compute_joints_visibility(
        smplh_model,
        world_to_camera,
        coco_joints_3d,
        coco_joints_2d,
        VISIBILITY_THRESHOLD,
        mask,
        show_preview=PREVIEW,
        preview_resolution=resolution,
    )

    h36m_joints_vis, h36m_joints_vis_flags = compute_joints_visibility(
        smplh_model,
        world_to_camera,
        h36m_joints_3d,
        h36m_joints_2d,
        VISIBILITY_THRESHOLD,
        mask,
        show_preview=PREVIEW,
        preview_resolution=resolution,
    )


    if PREVIEW:
        img = cv2.imread(format_img_path(identity, frame))

        smpl_img = img.copy()
        coco_img = img.copy()
        h36m_img = img.copy()
        draw_skeleton_2d(smpl_img, smpl_joints_2d, smpl_joints_vis, SMPL_CONNECTIVITY)
        draw_skeleton_2d(coco_img, coco_joints_2d, coco_joints_vis, COCO_CONNECTIVITY)
        draw_skeleton_2d(h36m_img, h36m_joints_2d, h36m_joints_vis, H36M_CONNECTIVITY)

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

    # Output the mask
    cv2.imwrite(format_output_mask_path(identity, frame), mask.astype(np.uint8) * 255)
    # Output the extra metadata
    extra_metadata = {
        "bbox": bbox,
        "smpl": {
            "joints_2d": smpl_joints_2d.tolist(),
            "joints_3d": smpl_joints_3d.tolist(),
            "joints_vis": smpl_joints_vis.tolist(),
            "joints_vis_flags": smpl_joints_vis_flags.tolist(),
        },
        "coco": {
            "joints_2d": coco_joints_2d.tolist(),
            "joints_3d": coco_joints_3d.tolist(),
            "joints_vis": coco_joints_vis.tolist(),
            "joints_vis_flags": coco_joints_vis_flags.tolist(),
        },
        "h36m": {
            "joints_2d": h36m_joints_2d.tolist(),
            "joints_3d": h36m_joints_3d.tolist(),
            "joints_vis": h36m_joints_vis.tolist(),
            "joints_vis_flags": h36m_joints_vis_flags.tolist(),
        },
    }
    with open(format_output_extra_metadata_path(identity, frame), "w") as f:
        json.dump(extra_metadata, f)


for identity, frame in tqdm(
    itertools.product(range(N_IDENTITIES), range(N_FRAMES_PER_IDENTITY)),
    total=N_IDENTITIES * N_FRAMES_PER_IDENTITY,
):
    process_identity_frame(SMPLH_MODEL, identity, frame)
