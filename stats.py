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

    print("Loading instances.json...")
    instances = json.load(open(os.path.join(args.dataset_dir, "instances.json"), "r"))
    n_instances = len(instances)

    # Average number of visible joints per instance
    smpl_n_visible_joints_avg = 0
    coco_n_visible_joints_avg = 0
    h36m_n_visible_joints_avg = 0

    # Visibility probability for each joint
    smpl_per_joint_visibility_p = np.zeros((N_SMPLH_JOINTS,))
    coco_per_joint_visibility_p = np.zeros((N_COCO_JOINTS,))
    h36m_per_joint_visibility_p = np.zeros((N_H36M_JOINTS,))

    for instance in tqdm(instances, desc="Parsing metadata"):
        extra_metadata_fp = os.path.join(args.dataset_dir, instance["extra_metadata"])
        extra_metadata = json.load(open(extra_metadata_fp, "r"))

        smpl_joints_vis = np.array(extra_metadata["joints"]["smpl"]["joints_vis"])
        coco_joints_vis = np.array(extra_metadata["joints"]["coco"]["joints_vis"])
        h36m_joints_vis = np.array(extra_metadata["joints"]["h36m"]["joints_vis"])

        smpl_n_visible_joints_avg += np.sum(smpl_joints_vis)
        coco_n_visible_joints_avg += np.sum(coco_joints_vis)
        h36m_n_visible_joints_avg += np.sum(h36m_joints_vis)

        smpl_per_joint_visibility_p += smpl_joints_vis
        coco_per_joint_visibility_p += coco_joints_vis
        h36m_per_joint_visibility_p += h36m_joints_vis

    smpl_n_visible_joints_avg = smpl_n_visible_joints_avg / n_instances
    coco_n_visible_joints_avg = coco_n_visible_joints_avg / n_instances
    h36m_n_visible_joints_avg = h36m_n_visible_joints_avg / n_instances

    smpl_per_joint_visibility_p = smpl_per_joint_visibility_p / n_instances
    coco_per_joint_visibility_p = coco_per_joint_visibility_p / n_instances
    h36m_per_joint_visibility_p = h36m_per_joint_visibility_p / n_instances

    print("Average number of visible joints per instance:")
    print(f"SMPL: {smpl_n_visible_joints_avg / N_SMPLH_JOINTS * 100:.2f}%")
    print(f"COCO: {coco_n_visible_joints_avg / N_COCO_JOINTS * 100:.2f}%")
    print(f"H36M: {h36m_n_visible_joints_avg / N_H36M_JOINTS * 100:.2f}%")

    if args.plot:
        plt.figure()
        plt.pie(
            [smpl_n_visible_joints_avg, N_SMPLH_JOINTS - smpl_n_visible_joints_avg],
            labels=["Visible", "Occluded"],
            autopct="%1.1f%%",
        )
        plt.title("SMPL average number of visible joints per instance")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.pie(
            [coco_n_visible_joints_avg, N_COCO_JOINTS - coco_n_visible_joints_avg],
            labels=["Visible", "Occluded"],
            autopct="%1.1f%%",
        )
        plt.title("COCO average number of visible joints per instance")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.pie(
            [h36m_n_visible_joints_avg, N_H36M_JOINTS - h36m_n_visible_joints_avg],
            labels=["Visible", "Occluded"],
            autopct="%1.1f%%",
        )
        plt.title("H3.6M average number of visible joints per instance")
        plt.tight_layout()
        plt.show()

    print("Visibility probability for each joint:")
    print("SMPL:")
    for i, vis_p in enumerate(smpl_per_joint_visibility_p):
        print(f"\t- {SMPLH_JOINT_NAMES[i]}: {vis_p * 100:.2f}%")
    print("- COCO:")
    for i, vis_p in enumerate(coco_per_joint_visibility_p):
        print(f"\t- {COCO_JOINT_NAMES[i]}: {vis_p * 100:.2f}%")
    print("- H3.6M:")
    for i, vis_p in enumerate(h36m_per_joint_visibility_p):
        print(f"\t- {H36M_JOINT_NAMES[i]}: {vis_p * 100:.2f}%")

    if args.plot:
        plt.figure()
        plt.bar(SMPLH_JOINT_NAMES, smpl_per_joint_visibility_p)
        plt.xticks(rotation=90)
        plt.title("Visibility probability for each joint (SMPL)")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.bar(COCO_JOINT_NAMES, coco_per_joint_visibility_p)
        plt.xticks(rotation=90)
        plt.title("Visibility probability for each joint (COCO)")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.bar(H36M_JOINT_NAMES, h36m_per_joint_visibility_p)
        plt.xticks(rotation=90)
        plt.title("Visibility probability for each joint (H3.6M)")
        plt.tight_layout()
        plt.show()

    stats = {
        "smpl_n_visible_joints_avg": smpl_n_visible_joints_avg,
        "coco_n_visible_joints_avg": coco_n_visible_joints_avg,
        "h36m_n_visible_joints_avg": h36m_n_visible_joints_avg,
        "smpl_per_joint_visibility_p": smpl_per_joint_visibility_p.tolist(),
        "coco_per_joint_visibility_p": coco_per_joint_visibility_p.tolist(),
        "h36m_per_joint_visibility_p": h36m_per_joint_visibility_p.tolist(),
    }

    json.dump(stats, open(args.output_file, "w"), indent=4)


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
        "--output_file",
        type=str,
        default="output/stats.json",
        help="Path to the output file.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the statistics.",
    )
    args = parser.parse_args()

    compute_stats(args)
