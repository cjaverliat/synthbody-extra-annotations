import argparse
import os
import json
import numpy as np
from constants import SMPLH_JOINT_NAMES, COCO_JOINT_NAMES, H36M_JOINT_NAMES
import matplotlib.pyplot as plt
from tqdm import tqdm

N_SMPLH_JOINTS = 52
N_COCO_JOINTS = 17
N_H36M_JOINTS = 17


def compute_stats(args):

    annotations = np.load(
        os.path.join(args.dataset_dir, f"annotations_{args.skeleton_type}.npy")
    )

    if args.skeleton_type == "smpl":
        n_joints = N_SMPLH_JOINTS
        joints_names = SMPLH_JOINT_NAMES
    elif args.skeleton_type == "coco":
        n_joints = N_COCO_JOINTS
        joints_names = COCO_JOINT_NAMES
    elif args.skeleton_type == "h36m":
        n_joints = N_H36M_JOINTS
        joints_names = H36M_JOINT_NAMES

    # Average number of visible joints per instance
    n_visible_joints_avg = 0

    # Visibility probability for each joint
    per_joint_visibility_p = np.zeros((n_joints,))

    for annotation in tqdm(annotations, desc="Parsing metadata"):
        joints_vis = np.array(annotation["joints_vis"])
        n_visible_joints_avg += np.sum(joints_vis) / n_joints
        per_joint_visibility_p += joints_vis

    n_visible_joints_avg = n_visible_joints_avg / len(annotations)
    per_joint_visibility_p = per_joint_visibility_p / len(annotations)

    print(
        f"Average number of visible joints per instance: {n_visible_joints_avg * 100:.2f}%"
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

    with open(
        os.path.joint(args.output_dir, f"stats_{args.skeleton_type}.json"), "w"
    ) as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate statistics about the SynthBody dataset."
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the dataset directory (path/to/synth_body/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--skeleton_type",
        type=str,
        default="coco",
        help="Type of skeleton to use for the joints annotations.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the statistics.",
    )
    args = parser.parse_args()

    if args.skeleton_type not in ["smpl", "coco", "h36m"]:
        raise ValueError(
            "Invalid skeleton type. Must be one of ['smpl', 'coco', 'h36']"
        )

    compute_stats(args)
