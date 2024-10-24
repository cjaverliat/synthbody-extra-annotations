import argparse

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
        "--preview_2d",
        action="store_true",
        help="Show a preview of the visibility computation.",
    )
    parser.add_argument(
        "--preview_3d",
        action="store_true",
        help="Show a preview of the visibility computation with 3D raycasts.",
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
