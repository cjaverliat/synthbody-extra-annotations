import os.path as osp
import cv2
import numpy as np

from pathlib import Path
import json
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

STATS_FILE = "F:/Documents/These/Datasets/SynthMoCap/data/synth_body_extras_stats.json"

VISIBLE_FLAG = 0
SELF_OCCLUDED_FLAG = 1
EXTERNAL_OCCLUDED_FLAG = 2
SELF_AND_EXTERNAL_OCCLUDED_FLAG = 3

SMPLH_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

stats = json.load(open(STATS_FILE, "r"))

n_samples = int(stats["n_samples"])

smpl_visible_joints = np.array(stats["smpl"]["visible"])
smpl_self_occluded_joints = np.array(stats["smpl"]["self_occluded"])
smpl_external_occluded_joints = np.array(stats["smpl"]["external_occluded"])
smpl_self_and_external_occluded_joints = np.array(stats["smpl"]["self_and_external_occluded"])

coco_visible_joints = np.array(stats["coco"]["visible"])
coco_self_occluded_joints = np.array(stats["coco"]["self_occluded"])
coco_external_occluded_joints = np.array(stats["coco"]["external_occluded"])
coco_self_and_external_occluded_joints = np.array(stats["coco"]["self_and_external_occluded"])

h36m_visible_joints = np.array(stats["h36m"]["visible"])
h36m_self_occluded_joints = np.array(stats["h36m"]["self_occluded"])
h36m_external_occluded_joints = np.array(stats["h36m"]["external_occluded"])
h36m_self_and_external_occluded_joints = np.array(stats["h36m"]["self_and_external_occluded"])

# Plot SMPL stats as pie charts per joint side by side
n_joints = smpl_visible_joints.shape[0]
n_cols = 13  # Number of columns for the subplot grid
n_rows = (n_joints + n_cols - 1) // n_cols  # Calculate number of rows needed

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
axes = axes.flatten()

labels = ['Visible', 'Self occluded', 'External occluded', 'Self and external occluded']
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

for joint_idx in range(n_joints):
    sizes = [
        smpl_visible_joints[joint_idx],
        smpl_self_occluded_joints[joint_idx],
        smpl_external_occluded_joints[joint_idx],
        smpl_self_and_external_occluded_joints[joint_idx]
    ]
    wedges, texts, autotexts = axes[joint_idx].pie(sizes, colors=colors, autopct='%1.1f%%', startangle=140)
    axes[joint_idx].set_title(f'{SMPLH_JOINT_NAMES[joint_idx]}')
    axes[joint_idx].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Hide any unused subplots
for idx in range(n_joints, len(axes)):
    axes[idx].axis('off')

# Add a common legend
fig.legend(wedges, labels=labels, loc='upper right', bbox_to_anchor=(1.1, 1))

plt.tight_layout()
plt.show()


