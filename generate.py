import os
import cv2
import numpy as np
from smpl_numpy import SMPL
import json
import itertools
from tqdm import tqdm
import argparse

from masks import generate_mask, generate_bbox_annotation
from joints import generate_joints_annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings


def generate_instance_extras(identity, frame):
    metadata_fp = os.path.join(
        args.dataset_dir,
        f"metadata_{identity:07d}_{frame:03d}.json",
    )

    if not os.path.exists(metadata_fp):
        return None

    formatted_suffix = f"{identity:07d}_{frame:03d}"

    mask = generate_mask(args.dataset_dir, identity, frame)
    bbox_annotation = generate_bbox_annotation(mask)
    joints_annotations = generate_joints_annotations(
        args.smplh_model,
        args.dataset_dir,
        identity,
        frame,
        args.visibility_threshold,
        mask,
        args.preview_2d,
        args.preview_3d,
    )

    extra_metadata = {
        "bbox": bbox_annotation,
        "joints": joints_annotations,
    }

    cv2.imwrite(
        os.path.join(args.dataset_dir, "extras", f"mask_{formatted_suffix}.png"),
        mask.astype(np.uint8) * 255,
    )

    json.dump(
        extra_metadata,
        open(
            os.path.join(
                args.dataset_dir,
                "extras",
                f"extra_metadata_{formatted_suffix}.json",
            ),
            "w",
        ),
        indent=4,
    )

    return {
        "identity": identity,
        "frame": frame,
        "image": f"img_{formatted_suffix}.jpg",
        "metadata": f"metadata_{formatted_suffix}.json",
        "extra_metadata": f"extras/extra_metadata_{formatted_suffix}.json",
        "segm": {
            "full": f"extras/mask_{formatted_suffix}.png",
            "beard": f"segm_beard_{formatted_suffix}.png",
            "eyebrows": f"segm_eyebrows_{formatted_suffix}.png",
            "eyelashes": f"segm_eyelashes_{formatted_suffix}.png",
            "facewear": f"segm_facewear_{formatted_suffix}.png",
            "glasses": f"segm_glasses_{formatted_suffix}.png",
            "hair": f"segm_hair_{formatted_suffix}.png",
            "headwear": f"segm_headwear_{formatted_suffix}.png",
            "parts": f"segm_parts_{formatted_suffix}.png",
        },
    }


def generate_extras(args):

    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory {args.dataset_dir} not found.")

    # Ensure the extras folder exists
    os.makedirs(os.path.join(args.dataset_dir, "extras"), exist_ok=True)

    instances = []

    if args.n_workers == 0:
        for identity, frame in tqdm(
            itertools.product(
                range(args.n_identities), range(args.n_frames_per_identity)
            ),
            total=args.n_identities * args.n_frames_per_identity,
        ):
            res = generate_instance_extras(identity, frame)
            if res is not None:
                instances.append(res)
    else:
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [
                executor.submit(generate_instance_extras, identity, frame)
                for identity, frame in itertools.product(
                    range(args.n_identities), range(args.n_frames_per_identity)
                )
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                res = future.result()
                if res is not None:
                    instances.append(res)

    json.dump(
        instances,
        open(os.path.join(args.dataset_dir, "instances.json"), "w"),
        indent=4,
    )


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
    generate_extras(args)
