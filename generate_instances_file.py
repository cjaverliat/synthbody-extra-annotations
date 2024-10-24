import os.path as osp
from tqdm import tqdm
import json
import argparse


def generate_instances_file(dataset_dir, output_file, n_instances=20000, n_frames=5):
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
                    "instance_id": i,
                    "frame_id": j,
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

    json.dump(instances, open(output_file, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instances.json for the SynthBody dataset."
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the dataset directory (path/to/synth_body/).",
    )
    parser.add_argument("output_file", type=str, help="Path to the output JSON file.")
    parser.add_argument(
        "--n_instances", type=int, default=20000, help="Number of instances to process."
    )
    parser.add_argument(
        "--n_frames", type=int, default=5, help="Number of frames per instance."
    )

    args = parser.parse_args()

    generate_instances_file(
        args.dataset_dir, args.output_file, args.n_instances, args.n_frames
    )
