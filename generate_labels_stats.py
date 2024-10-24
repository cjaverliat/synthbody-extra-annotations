import os.path as osp
import numpy as np
import json
import itertools
from tqdm import tqdm

DATA_DIR = "F:/Documents/These/Datasets/SynthMoCap/data/synth_body_extras/"
OUTPUT_FILE = "F:/Documents/These/Datasets/SynthMoCap/data/synth_body_extras_stats.json"
N_IDENTITIES = 20000
N_FRAMES_PER_IDENTITY = 5

VISIBLE_FLAG = 0
SELF_OCCLUDED_FLAG = 1
EXTERNAL_OCCLUDED_FLAG = 2
SELF_AND_EXTERNAL_OCCLUDED_FLAG = 3

def format_extra_metadata_path(identity, frame):
    return osp.join(DATA_DIR, f"extra_metadata_{identity:07d}_{frame:03d}.json")

smpl_visible_joints = np.zeros((52))
smpl_self_occluded_joints = np.zeros((52))
smpl_external_occluded_joints = np.zeros((52))
smpl_self_and_external_occluded_joints = np.zeros((52))

coco_visible_joints = np.zeros((17))
coco_self_occluded_joints = np.zeros((17))
coco_external_occluded_joints = np.zeros((17))
coco_self_and_external_occluded_joints = np.zeros((17))

h36m_visible_joints = np.zeros((17))
h36m_self_occluded_joints = np.zeros((17))
h36m_external_occluded_joints = np.zeros((17))
h36m_self_and_external_occluded_joints = np.zeros((17))

n_samples = 0

for identity, frame in tqdm(
    itertools.product(range(N_IDENTITIES), range(N_FRAMES_PER_IDENTITY)),
    total=N_IDENTITIES * N_FRAMES_PER_IDENTITY,
):
    extra_metadata_fp = format_extra_metadata_path(identity, frame)

    if not osp.exists(extra_metadata_fp):
        print(f"Extra metadata file {extra_metadata_fp} does not exist.")
        continue

    n_samples += 1

    extra_metadata = json.load(open(extra_metadata_fp, "r"))
    smpl_joints_vis_flags = np.array(extra_metadata["smpl"]["joints_vis_flags"])
    smpl_visible_joints += smpl_joints_vis_flags == VISIBLE_FLAG
    smpl_self_occluded_joints += smpl_joints_vis_flags == SELF_OCCLUDED_FLAG
    smpl_external_occluded_joints += smpl_joints_vis_flags == EXTERNAL_OCCLUDED_FLAG
    smpl_self_and_external_occluded_joints += smpl_joints_vis_flags == SELF_AND_EXTERNAL_OCCLUDED_FLAG

    coco_joints_vis_flags = np.array(extra_metadata["coco"]["joints_vis_flags"])
    coco_visible_joints += coco_joints_vis_flags == VISIBLE_FLAG
    coco_self_occluded_joints += coco_joints_vis_flags == SELF_OCCLUDED_FLAG
    coco_external_occluded_joints += coco_joints_vis_flags == EXTERNAL_OCCLUDED_FLAG
    coco_self_and_external_occluded_joints += coco_joints_vis_flags == SELF_AND_EXTERNAL_OCCLUDED_FLAG

    h36m_joints_vis_flags = np.array(extra_metadata["h36m"]["joints_vis_flags"])
    h36m_visible_joints += h36m_joints_vis_flags == VISIBLE_FLAG
    h36m_self_occluded_joints += h36m_joints_vis_flags == SELF_OCCLUDED_FLAG
    h36m_external_occluded_joints += h36m_joints_vis_flags == EXTERNAL_OCCLUDED_FLAG
    h36m_self_and_external_occluded_joints += h36m_joints_vis_flags == SELF_AND_EXTERNAL_OCCLUDED_FLAG

# Save the stats
stats = {
    "n_samples": n_samples,
    "smpl": {
        "visible": smpl_visible_joints.tolist(),
        "self_occluded": smpl_self_occluded_joints.tolist(),
        "external_occluded": smpl_external_occluded_joints.tolist(),
        "self_and_external_occluded": smpl_self_and_external_occluded_joints.tolist(),
    },
    "coco": {
        "visible": coco_visible_joints.tolist(),
        "self_occluded": coco_self_occluded_joints.tolist(),
        "external_occluded": coco_external_occluded_joints.tolist(),
        "self_and_external_occluded": coco_self_and_external_occluded_joints.tolist(),
    },
    "h36m": {
        "visible": h36m_visible_joints.tolist(),
        "self_occluded": h36m_self_occluded_joints.tolist(),
        "external_occluded": h36m_external_occluded_joints.tolist(),
        "self_and_external_occluded": h36m_self_and_external_occluded_joints.tolist(),
    },
}

json.dump(stats, open(OUTPUT_FILE, "w"))
