import os
from smpl_numpy import SMPL
import json
import itertools
from tqdm import tqdm
import argparse
import numpy as np
import cv2
import pickle

from bbox import generate_mask, generate_bbox_annotation_from_mask
from joints import generate_joints_annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings


def generate_instance_annotations(args, identity, frame):

    formatted_suffix = f"{identity:07d}_{frame:03d}"

    metadata_fp = os.path.join(
        args.dataset_dir,
        f"metadata_{formatted_suffix}.json",
    )

    if not os.path.exists(metadata_fp):
        return None

    with open(metadata_fp, "r") as f:
        metadata = json.load(f)

    preview_img = None

    if args.preview_2d:
        preview_img_fp = os.path.join(
            args.dataset_dir,
            f"img_{formatted_suffix}.jpg",
        )
        preview_img = cv2.imread(preview_img_fp)

    segmentation_mask = generate_mask(args.dataset_dir, identity, frame)

    body_identity = np.array(metadata["body_identity"])
    body_pose = np.array(metadata["pose"])
    body_translation = np.array(metadata["translation"])
    camera_w2c = np.array(metadata["camera"]["world_to_camera"])
    camera_K = np.array(metadata["camera"]["camera_to_image"])
    camera_resolution = np.array(metadata["camera"]["resolution"])

    bbox_annotation = generate_bbox_annotation_from_mask(segmentation_mask)

    joints_3d, joints_2d, joints_vis = generate_joints_annotations(
        smplh_model=args.smplh_model,
        skeleton_type=args.skeleton_type,
        identity=identity,
        frame=frame,
        segmentation_mask=segmentation_mask,
        body_identity=body_identity,
        body_pose=body_pose,
        body_translation=body_translation,
        camera_w2c=camera_w2c,
        camera_K=camera_K,
        camera_resolution=camera_resolution,
        visibility_threshold=args.visibility_threshold,
        preview_2d_img=preview_img,
        preview_2d=args.preview_2d,
        preview_3d=args.preview_3d,
    )

    instance_annotations = {
        "image": f"img_{formatted_suffix}.jpg",
        "identity": identity,
        "frame": frame,
        "bbox": bbox_annotation,
        "joints_3d": joints_3d,
        "joints_2d": joints_2d,
        "joints_vis": joints_vis,
        "camera_w2c": camera_w2c,
        "camera_K": camera_K,
        "camera_resolution": camera_resolution,
    }

    return instance_annotations


def generate_annotations(args):

    instances = []

    if args.n_workers == 0:
        for identity, frame in tqdm(
            itertools.product(
                range(args.n_identities), range(args.n_frames_per_identity)
            ),
            total=args.n_identities * args.n_frames_per_identity,
        ):
            instance_annotations = generate_instance_annotations(args, identity, frame)
            if instance_annotations is not None:
                instances.append(instance_annotations)
    else:
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [
                executor.submit(generate_instance_annotations, args, identity, frame)
                for identity, frame in itertools.product(
                    range(args.n_identities), range(args.n_frames_per_identity)
                )
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                instance_annotations = future.result()
                if instance_annotations is not None:
                    instances.append(instance_annotations)

    annotations_fp = os.path.join(
        args.output_dir, f"annotations_{args.skeleton_type}.pkl"
    )
    annotations = {
        "skeleton_type": args.skeleton_type,
        "visibility_threshold": args.visibility_threshold,
        "instances": instances,
    }

    with open(annotations_fp, "wb") as f:
        pickle.dump(annotations, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate extra annotations for the SynthBody dataset."
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the dataset directory (path/to/synth_body/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        help="Path to the output directory where the annotations will be saved.",
    )
    parser.add_argument(
        "--n_identities",
        type=int,
        default=20000,
        help="Number of identities to process.",
    )
    parser.add_argument(
        "--n_frames_per_identity",
        type=int,
        default=5,
        help="Number of frames per identity.",
    )
    parser.add_argument(
        "--preview_2d",
        action="store_true",
        help="Show the SMPL, COCO and H3.6M skeletons along with the visibility.",
    )
    parser.add_argument(
        "--preview_3d",
        action="store_true",
        help="Show a raycast scene used for computing the joints visibility.",
    )
    parser.add_argument(
        "--smplh_model",
        type=str,
        default="smplh/SMPLH_NEUTRAL.npz",
        help="Path to the SMPLH model.",
    )
    parser.add_argument(
        "--skeleton_type",
        type=str,
        default="coco",
        help="Type of skeleton to use for the joints annotations.",
    )
    parser.add_argument(
        "--visibility_threshold",
        type=float,
        default=0.3,
        help="Visibility threshold for the joints (in centimeters).",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=0,
        help="Number of workers to use for parallel processing.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory {args.dataset_dir} not found.")

    if args.skeleton_type not in ["smpl", "coco", "h36m"]:
        raise ValueError(
            f"Invalid skeleton type {args.skeleton_type}. Must be one of ['smpl', 'coco', 'h36m']."
        )

    if args.n_workers > 0 and (args.preview_2d or args.preview_3d):
        warnings.warn(
            "Disabling previews when using multiple workers. Set n_workers to 0 to enable previews."
        )
        args.preview_2d = False
        args.preview_3d = False

    # Print a summary of the arguments
    print("Extra data will be generated with the following arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    args.smplh_model = SMPL(args.smplh_model)

    os.makedirs(args.output_dir, exist_ok=True)

    generate_annotations(args)
