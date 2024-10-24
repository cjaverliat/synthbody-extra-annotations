# SynthBody Extra Annotations

This repository contains the code to generate extra annotations for the [SynthBody dataset](https://github.com/microsoft/SynthMoCap). In particular, we generate the 2D and 3D joints positions for the SMPL, COCO and H3.6M skeletons formats along with a visibility flag for each joint.

## Download the extra annotations

The extra annotations are available for download [here]().

## Generate the extra annotations

1. Start by downloading the SynthBody dataset from the [SynthMoCap repository](https://github.com/microsoft/SynthMoCap/blob/main/DATASETS.md).

2. Clone this repository:
```sh
git clone https://github.com/cjaverliat/synthbody-extra-annotations.git
cd synthbody-extra-annotations
```

3. Install the required dependencies in a virtual environment:
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Generate the extra annotations for the SynthBody dataset using the following command:
```sh
python generate.py path/to/SynthMoCap/synth_body/ --smplh_model path/to/smplh/model_neutral.npz --n_workers 8
```
This will result in the following files:

```
synth_body/
├── ...
├── instances.json
└── extras/
    ├── extra_metadata_XXXXXXX_XXX.json
    ├── ...
    └── mask_XXXXXXX_XXX.png
```

## Preview the generated extra annotations

You can preview the generated extra annotations using the following command:
```sh
python generate.py path/to/SynthMoCap/synth_body/ --smplh_model path/to/smplh/model_neutral.npz --preview_2d --preview_3d
```
This will display a 3D scene with the raycast used to generate the visibility flag for each joint and a 2D preview of the SMPL, COCO and H3.6M skeletons. The visible joints are colored in green and the occluded joints are colored in red.

![2D preview of SMPL, COCO and H3.6M skeletons. Visible joints are green. Occluded joints are red.](img/preview_2d.png)