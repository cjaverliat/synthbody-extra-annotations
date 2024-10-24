import json
from tqdm import tqdm
import os.path as osp


def generate_instances_list(dataset_dir, n_instances=20000, n_frames=5):
    instances = []
    pbar = tqdm(total=n_instances * n_frames)

    for i in range(n_instances):
        for j in range(n_frames):
            pbar.update(1)
            metadata_fp = osp.join(
                dataset_dir,
                f"metadata_{i:07d}_{j:03d}.json",
            )

            if not osp.exists(metadata_fp):
                continue

            formatted_suffix = f"{i:07d}_{j:03d}"

            instances.append(
                {
                    "identity": i,
                    "frame": j,
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
            )

    return instances
