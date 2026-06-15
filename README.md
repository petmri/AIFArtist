# AIFArtist

AIFArtist is a napari desktop app for manual arterial input function annotation on 4D MRI NIfTI data. It is designed for high-volume review sessions where multiple raters need to draw a 3D ROI, inspect the mean signal-intensity curve over time, save a BIDS-style derivative, and move directly to the next image.

## Features

- Load one or more BIDS-compliant 4D `desc-hmc_DCE.nii` or `desc-hmc_DCE.nii.gz` files, a directory tree, or a manifest file.
- Draw and edit a 3D ROI in napari using an editable labels layer.
- Preview the live mean intensity curve across timepoints for the selected ROI, with separate curves for each painted label so label 1 and label 2 can be compared directly, plus optional graphs normalized to the first or second timepoint.
- Save outputs into a BIDS-style derivatives layout with the rater identifier embedded in the filenames, while preserving source entities such as `task`, `acq`, and `run` to avoid filename collisions.
- Resume work efficiently by auto-jumping to the first unreviewed image for the current rater, prefetching the next image so Save and Next can usually switch immediately, and letting raters flag poor AIFs or missing baselines to remove them from the queue.

## Install

Use the existing virtual environment or create a new one, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

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

By default, the queue skips images that already have a saved ROI for the current rater, and it always skips images that the current rater has flagged. Use `--include-completed` to reopen completed cases:

```bash
python aif_artist.py /path/to/bids_dataset --rater AB --include-completed
```

By default, outputs are written to `/media/network_mriphysics/dce_bids/derivatives`. Use `--output-root` to override that location, or `--write-sidecars` if you also want the ROI timeseries TSV and metadata JSON files.

## Outputs

By default, outputs are written under `/media/network_mriphysics/dce_bids/derivatives`. For each annotated image, the tool writes:

- `*_desc-raterXX_label-AIF_mask.nii.gz`: saved 3D ROI mask.
- `desc-raterXX_flags.csv`: per-rater flag log with `img` and `reason` columns for skipped poor-AIF or missing-baseline cases.

If `--write-sidecars` is provided, the tool also writes:

- `*_desc-raterXX_label-AIF_timeseries.tsv`: mean signal over time within the ROI.
- `*_desc-raterXX_label-AIF_timeseries.json`: metadata including rater, source image, shape, and voxel count.

The derivative root also gets a `dataset_description.json` file.

## Workflow Notes

- The app skips images that already have a saved ROI for the current rater unless `--include-completed` is provided, and it always skips images that rater has flagged.
- The app opens at the first remaining queue item unless `--include-completed` is provided, in which case `--start-index` applies directly to the full queue.
- If a saved ROI already exists for that rater and image, it is loaded automatically for editing.
- Use the `Flag and Skip` button to mark the current image as `Poor AIF` or `Missing baseline`, append that decision to the per-rater flags CSV, and move straight to the next queue item.
- The queue discovery step filters down to 4D `desc-hmc_DCE.nii` and `desc-hmc_DCE.nii.gz` images anywhere under the provided inputs, including derivative datasets.

## Controls Reference

### Viewer Controls

- Use napari's built-in 2D/3D view toggle to switch between slice view and volume view. The first 2D entry defaults to a coronal slice view, and the default 3D camera is coronal as well.
- In 2D view, plain `scroll` steps through slices.
- In 2D or 3D view, `Ctrl+scroll` steps through time frames. Wheel up moves forward in time, and wheel down moves backward.
- In 3D view, `Shift+scroll` adjusts the upper window limit.
- In 3D view, `Alt+scroll` adjusts the lower window limit.
- In 3D view, `Ctrl+Shift+scroll` zooms.
- In 2D view, `right-click drag` on the active ROI labels layer temporarily switches to erase so you can remove voxels without changing the current paint mode.
- ROI painting, fill, and label-number selection otherwise use the standard napari labels-layer controls.

### Dock Controls

- `Display Frame` slider and spinbox change the displayed time frame directly.
- `Window / Level` low and high spinboxes and sliders adjust image contrast numerically or by dragging.
- `Auto` under `Window / Level` resets the contrast limits from the currently displayed frame.
- `Show normalized-to-first-point graph` toggles the extra curve plot normalized to the first timepoint.
- `Show normalized-to-second-point graph` toggles the extra curve plot normalized to the second timepoint.
- `Previous` loads the previous image in the queue.
- `Clear ROI` removes the current ROI labels from the image.
- `Skip` advances to the next image without saving.
- `Flag and Skip` records a `Poor AIF` or `Missing baseline` flag for the current image and advances.
- `Save ROI and Next` saves the ROI and advances. The keyboard shortcut is `Ctrl+Enter`.