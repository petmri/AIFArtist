# AIFArtist

AIFArtist is a napari desktop app for manual arterial input function annotation on 4D MRI NIfTI data. It is designed for high-volume review sessions where multiple raters need to draw a 3D ROI, inspect the mean signal-intensity curve over time, save a BIDS-style derivative, and move directly to the next image.

## Features

- Load one or more BIDS-compliant 4D `desc-hmc_DCE.nii` or `desc-hmc_DCE.nii.gz` files, a directory tree, or a manifest file.
- Draw and edit a 3D ROI in napari using an editable labels layer.
- Preview the live mean intensity curve across timepoints for the selected ROI.
- Save outputs into a BIDS-style `derivatives/aifartist` layout with the rater identifier embedded in the filenames and metadata, while preserving source entities such as `task`, `acq`, and `run` to avoid filename collisions.
- Resume work efficiently by auto-jumping to the first unsaved image for the current rater and auto-advancing after each save.

## Install

Use the existing virtual environment or create a new one, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

Launch against one or more files or folders:

```bash
python aif_artist.py /path/to/bids_dataset --rater AB
```

Or use a manifest:

```bash
python aif_artist.py --manifest image_list.tsv --rater AB
```

The manifest may be a plain text file with one path per line, or a CSV/TSV with one of these columns: `path`, `image`, `image_path`, or `nifti`.

By default, the queue skips images that already have a saved ROI for the current rater. Use `--include-completed` to reopen completed cases:

```bash
python aif_artist.py /path/to/bids_dataset --rater AB --include-completed
```

## Outputs

By default, outputs are written near the input dataset under `derivatives/aifartist`. For each annotated image, the tool writes:

- `*_desc-raterXX_label-aif_mask.nii.gz`: saved 3D ROI mask.
- `*_desc-raterXX_label-aif_timeseries.tsv`: mean signal over time within the ROI.
- `*_desc-raterXX_label-aif_timeseries.json`: metadata including rater, source image, shape, and voxel count.

The derivative root also gets a `dataset_description.json` file.

## Workflow Notes

- The app skips images that already have a saved ROI for the current rater unless `--include-completed` is provided.
- The app opens at the first remaining queue item unless `--include-completed` is provided, in which case `--start-index` applies directly to the full queue.
- If a saved ROI already exists for that rater and image, it is loaded automatically for editing.
- The queue discovery step filters down to 4D `desc-hmc_DCE.nii` and `desc-hmc_DCE.nii.gz` images anywhere under the provided inputs, including derivative datasets.
- In both 2D and 3D views, use `Ctrl+scroll` to step through time frames. In 3D view, use `Shift+scroll` to adjust the upper window limit, `Alt+scroll` to adjust the lower window limit, and `Shift+Ctrl+scroll` to zoom. In 2D view, plain scroll still steps through slices.