# SynthBody Extra Annotations

This repository contains the code to generate extra annotations for the SynthBody dataset. In particular, we generate the 2D and 3D joints positions for the SMPL, COCO and H3.6M skeletons formats along with a visibility flag for each joint.

## Generate the extra annotations

Generate the extra annotations for the SynthBody dataset.

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

## Preview the extra annotations

```sh
python generate.py path/to/SynthMoCap/synth_body/ --smplh_model path/to/smplh/model_neutral.npz --preview_2d --preview_3d
```

![2D preview of SMPL, COCO and H3.6M skeletons. Visible joints are green. Occluded joints are red.](img/preview_2d.png)