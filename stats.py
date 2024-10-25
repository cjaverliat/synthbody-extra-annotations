import argparse
import os
import json
import numpy as np
import pickle
from constants import SMPLH_JOINT_NAMES, COCO_JOINT_NAMES, H36M_JOINT_NAMES
import matplotlib.pyplot as plt
from tqdm import tqdm

N_SMPLH_JOINTS = 52
N_COCO_JOINTS = 17
N_H36M_JOINTS = 17


def compute_stats(args):

    with open(args.annotations_file, "rb") as f:
        annotations = pickle.load(f)

    instances_annotations = annotations["instances"]
    skeleton_type = annotations["skeleton_type"]

    if skeleton_type == "smpl":
        n_joints = N_SMPLH_JOINTS
        joints_names = SMPLH_JOINT_NAMES
    elif skeleton_type == "coco":
        n_joints = N_COCO_JOINTS
        joints_names = COCO_JOINT_NAMES
    elif skeleton_type == "h36m":
        n_joints = N_H36M_JOINTS
        joints_names = H36M_JOINT_NAMES

    # Average number of visible joints per instance
    n_visible_joints_avg = 0

    # Visibility probability for each joint
    per_joint_visibility_p = np.zeros((n_joints,))

    for instance_annotation in tqdm(instances_annotations, desc="Parsing metadata"):
        joints_vis = instance_annotation["joints_vis"]
        n_visible_joints_avg += np.sum(joints_vis)
        per_joint_visibility_p += joints_vis

    n_visible_joints_avg = n_visible_joints_avg / len(instances_annotations)
    per_joint_visibility_p = per_joint_visibility_p / len(instances_annotations)

    print(
        f"Average number of visible joints per instance: {n_visible_joints_avg:.2f} ({n_visible_joints_avg/n_joints*100:.2f}%)"
    )

    if args.plot:
        plt.figure()
        plt.pie(
            [n_visible_joints_avg, n_joints - n_visible_joints_avg],
            labels=["Visible", "Occluded"],
            autopct="%1.1f%%",
        )
        plt.title("Average number of visible joints per instance")
        plt.tight_layout()
        plt.show()

    print("Visibility probability for each joint:")
    for i, vis_p in enumerate(per_joint_visibility_p):
        print(f"- {joints_names[i]}: {vis_p * 100:.2f}%")

    if args.plot:
        plt.figure()
        plt.bar(joints_names, per_joint_visibility_p)
        plt.xticks(rotation=90)
        plt.title("Visibility probability for each joint (SMPL)")
        plt.tight_layout()
        plt.show()

    stats = {
        "n_visible_joints_avg": n_visible_joints_avg,
        "per_joint_visibility_p": per_joint_visibility_p.tolist(),
    }

    with open(os.path.join(args.output_dir, f"stats_{skeleton_type}.json"), "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate statistics about the SynthBody dataset."
    )
    parser.add_argument(
        "annotations_file",
        type=str,
        help="Path to the annotations file (pickle format).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the statistics.",
    )
    args = parser.parse_args()

    compute_stats(args)
