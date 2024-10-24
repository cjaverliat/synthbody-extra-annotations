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
import argparse

from masks import generate_mask, generate_bbox_annotation
from joints import generate_joints_annotations
from instances import generate_instances_list


def format_output_mask_path(dataset_dir, identity, frame):
    return osp.join(dataset_dir, "extras", f"mask_{identity:07d}_{frame:03d}.png")


def format_output_extra_metadata_path(dataset_dir, identity, frame):
    return osp.join(
        dataset_dir, "extras", f"extra_metadata_{identity:07d}_{frame:03d}.json"
    )


def format_img_path(dataset_dir, identity, frame):
    return osp.join(dataset_dir, f"img_{identity:07d}_{frame:03d}.jpg")


def format_metadata_path(dataset_dir, identity, frame):
    return osp.join(dataset_dir, f"metadata_{identity:07d}_{frame:03d}.json")


def generate_extras(args):

    instances_list = generate_instances_list(
        args.dataset_dir,
        args.n_identities,
        args.n_frames_per_identity,
    )

    json.dump(
        instances_list,
        open(osp.join(args.dataset_dir, "instances.json"), "w"),
        indent=4,
    )

    for instance in tqdm(instances_list):
        identity = instance["identity"]
        frame = instance["frame"]
        mask = generate_mask(args.dataset_dir, identity, frame)
        bbox_annotation = generate_bbox_annotation(mask)
        joints_annotations = generate_joints_annotations(
            args.smplh_model,
            args.dataset_dir,
            identity,
            frame,
            args.visibility_threshold,
            mask,
            args.preview,
        )

        cv2.imwrite(
            format_output_mask_path(args.dataset_dir, identity, frame),
            mask.astype(np.uint8) * 255,
        )

        extra_metadata = {
            "bbox": bbox_annotation,
            "joints": joints_annotations,
        }
        json.dump(
            extra_metadata,
            open(
                format_output_extra_metadata_path(args.dataset_dir, identity, frame),
                "w",
            ),
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances.json for the SynthBody dataset."
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the dataset directory (path/to/synth_body/).",
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
        "--preview",
        action="store_true",
        help="Show a preview of the visibility computation.",
    )
    parser.add_argument(
        "--smplh_model",
        type=str,
        default="smplh/SMPLH_NEUTRAL.npz",
        help="Path to the SMPLH model.",
    )
    parser.add_argument(
        "--visibility_threshold",
        type=float,
        default=0.3,
        help="Visibility threshold for the joints (in centimeters).",
    )

    args = parser.parse_args()

    # Print a summary of the arguments
    print("Extra data will be generated with the following arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    args.smplh_model = SMPL(args.smplh_model)
    generate_extras(args)
