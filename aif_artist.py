from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import napari
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.layers import Image, Labels
from qtpy.QtCore import QObject, QRunnable, QThreadPool, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)


SOURCE_IMAGE_SUFFIXES = ("desc-hmc_DCE.nii", "desc-hmc_DCE.nii.gz")
MASK_LABEL_VARIANTS = ("AIF", "aif")
FLAG_REASON_OPTIONS = ("Poor AIF", "Missing baseline")
DEFAULT_2D_ORDER = (1, 2, 0)
DEFAULT_OUTPUT_ROOT = Path("/media/network_mriphysics/dce_bids/derivatives/AIFArtist")


def normalize_2d_order(order: Sequence[int]) -> tuple[int, int, int]:
    normalized = tuple(int(axis) for axis in order)
    if len(normalized) == 3 and sorted(normalized) == [0, 1, 2]:
        return normalized
    return DEFAULT_2D_ORDER


def event_modifier_flags(event) -> set[str]:
    flags: set[str] = set()
    for modifier in getattr(event, "modifiers", ()) or ():
        for candidate in (modifier, getattr(modifier, "name", None)):
            if candidate is None:
                continue
            text = str(candidate).casefold()
            if "control" in text or "ctrl" in text:
                flags.add("control")
            if "shift" in text:
                flags.add("shift")
            if "alt" in text or "option" in text:
                flags.add("alt")
            if "meta" in text or "super" in text or "command" in text or text == "cmd":
                flags.add("meta")
    return flags


@dataclass(slots=True)
class ImageRecord:
    image_path: Path
    subject: str
    session: str | None
    stem: str
    entity_stem: str
    datatype: str


@dataclass(slots=True)
class SaveRequest:
    record_stem: str
    image_path: str
    rater_name: str
    image_shape: tuple[int, ...]
    mask: np.ndarray
    affine: np.ndarray
    header: nib.Nifti1Header
    curve: np.ndarray
    mask_path: Path
    tsv_path: Path
    json_path: Path
    write_sidecars: bool = False


@dataclass(slots=True)
class LoadedRecord:
    record: ImageRecord
    source_image: nib.Nifti1Image
    data: np.ndarray
    spacing: tuple[float, float, float]
    intensity_bounds: tuple[float, float]
    source_to_display: np.ndarray
    display_to_source: np.ndarray
    default_frame: int
    existing_mask: np.ndarray | None


class SaveTaskSignals(QObject):
    finished = Signal(str)
    failed = Signal(str, str)


class SaveTask(QRunnable):
    def __init__(self, request: SaveRequest) -> None:
        super().__init__()
        self.request = request
        self.signals = SaveTaskSignals()

    def run(self) -> None:
        try:
            write_aif_outputs(self.request)
        except Exception as exc:  # pragma: no cover - exercised through runtime worker path
            self.signals.failed.emit(self.request.record_stem, str(exc))
            return
        self.signals.finished.emit(self.request.record_stem)


class LoadTaskSignals(QObject):
    finished = Signal(int, int, object)
    failed = Signal(int, int, str, str)


class LoadTask(QRunnable):
    def __init__(
        self,
        request_id: int,
        index: int,
        record: ImageRecord,
        output_root: Path,
        rater_name: str,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.index = index
        self.record = record
        self.output_root = output_root
        self.rater_name = rater_name
        self.signals = LoadTaskSignals()

    def run(self) -> None:
        try:
            loaded_record = prepare_record_for_display(
                self.record,
                self.output_root,
                self.rater_name,
            )
        except Exception as exc:  # pragma: no cover - exercised through runtime worker path
            self.signals.failed.emit(
                self.request_id,
                self.index,
                self.record.image_path.name,
                str(exc),
            )
            return
        self.signals.finished.emit(self.request_id, self.index, loaded_record)


class CurveCanvas(FigureCanvasQTAgg):
    def __init__(self, title: str = "ROI Signal Intensity", y_label: str = "Mean intensity") -> None:
        figure = Figure(figsize=(5, 3), tight_layout=True)
        self.axes = figure.add_subplot(111)
        super().__init__(figure)
        self.title = title
        self.y_label = y_label
        self.axes.set_title(self.title)
        self.axes.set_xlabel("Timepoint")
        self.axes.set_ylabel(self.y_label)
        self.axes.grid(True, alpha=0.3)
        self.setMinimumWidth(0)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.updateGeometry()

    def update_curve(
        self,
        curves: dict[int, np.ndarray] | None,
        curve_colors: dict[int, tuple[float, float, float, float]] | None = None,
        empty_message: str = "Paint an AIF ROI to preview the mean time curve.",
    ) -> None:
        self.axes.clear()
        self.axes.set_title(self.title)
        self.axes.set_xlabel("Timepoint")
        self.axes.set_ylabel(self.y_label)
        self.axes.grid(True, alpha=0.3)
        if not curves:
            self.axes.text(
                0.5,
                0.5,
                empty_message,
                ha="center",
                va="center",
                transform=self.axes.transAxes,
            )
        else:
            for label_value, curve in sorted(curves.items()):
                color = None if curve_colors is None else curve_colors.get(label_value)
                timepoints = np.arange(curve.size)
                self.axes.plot(
                    timepoints,
                    curve,
                    linewidth=2,
                    color=color,
                    label=f"Label {label_value}",
                )
                self.axes.scatter(timepoints, curve, s=18, color=color)
            if len(curves) > 1:
                self.axes.legend(loc="best")
        self.draw_idle()


def prepare_record_for_display(
    record: ImageRecord,
    output_root: Path,
    rater_name: str,
) -> LoadedRecord:
    source_nifti = nib.load(str(record.image_path))
    display_nifti = nib.as_closest_canonical(source_nifti)
    data = np.asarray(display_nifti.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image for {record.image_path.name}, got shape {data.shape}.")

    spacing = extract_spatial_scale(display_nifti)
    intensity_bounds = compute_intensity_bounds(data)
    source_ornt = io_orientation(source_nifti.affine)
    display_ornt = io_orientation(display_nifti.affine)
    source_to_display = ornt_transform(source_ornt, display_ornt)
    display_to_source = ornt_transform(display_ornt, source_ornt)
    default_frame = infer_max_increase_frame(data)

    existing_mask = None
    mask_path = existing_mask_path_for_record(output_root, record, rater_name)
    if mask_path.exists():
        mask = np.asarray(nib.load(str(mask_path)).get_fdata(), dtype=np.uint8)
        mask = reorient_spatial_array(mask, source_to_display)
        if mask.shape != data.shape[:3]:
            raise ValueError(
                f"Existing ROI shape {mask.shape} does not match image shape {data.shape[:3]}."
            )
        existing_mask = np.asarray(mask, dtype=np.uint8)

    return LoadedRecord(
        record=record,
        source_image=source_nifti,
        data=data,
        spacing=spacing,
        intensity_bounds=intensity_bounds,
        source_to_display=source_to_display,
        display_to_source=display_to_source,
        default_frame=default_frame,
        existing_mask=existing_mask,
    )


def aifartist_mouse_wheel(viewer: napari.Viewer, event) -> None:
    signed_delta = float(event.delta[1]) if event.native.inverted() else -float(event.delta[1])
    widget = getattr(viewer, "_aifartist_widget", None)
    modifier_flags = event_modifier_flags(event)

    if viewer.dims.ndisplay == 2 and "control" not in modifier_flags:
        hidden_axes = viewer.dims.not_displayed
        if not hidden_axes:
            return

        hidden_axis = hidden_axes[0]
        viewer.dims._scroll_progress += signed_delta
        while abs(viewer.dims._scroll_progress) >= 1:
            step_offset = -1 if viewer.dims._scroll_progress < 0 else 1
            viewer.dims.set_current_step(
                hidden_axis,
                viewer.dims.current_step[hidden_axis] + step_offset,
            )
            viewer.dims._scroll_progress -= step_offset
        event.handled = True
        return

    if viewer.dims.ndisplay == 3 and "shift" in modifier_flags and "control" not in modifier_flags and "alt" not in modifier_flags:
        if widget is not None:
            widget.adjust_window_limit_from_scroll("high", signed_delta)
            event.handled = True
        return

    if viewer.dims.ndisplay == 3 and "alt" in modifier_flags and "control" not in modifier_flags and "shift" not in modifier_flags:
        if widget is not None:
            widget.adjust_window_limit_from_scroll("low", signed_delta)
            event.handled = True
        return

    if viewer.dims.ndisplay in (2, 3) and "control" in modifier_flags and "shift" not in modifier_flags and "alt" not in modifier_flags:
        if widget is not None:
            widget.step_frame_from_scroll(signed_delta)
            event.handled = True
        return

    if "control" in modifier_flags and "shift" in modifier_flags:
        viewer.camera.zoom *= 1.1 ** signed_delta
        event.handled = True


def aifartist_right_click_erase(viewer: napari.Viewer, event):
    widget = getattr(viewer, "_aifartist_widget", None)
    if widget is None or widget.labels_layer is None:
        return
    if viewer.dims.ndisplay != 2 or event.button != 2 or event.modifiers:
        return
    if viewer.layers.selection.active is not widget.labels_layer:
        return

    previous_mode = widget.labels_layer.mode
    widget.labels_layer.mode = "erase"
    event.handled = True
    yield

    while event.type == "mouse_move":
        event.handled = True
        yield

    if widget.labels_layer is not None:
        if viewer.dims.ndisplay == 2:
            widget.labels_layer.mode = previous_mode
        else:
            widget._sync_roi_interaction_mode()


class AIFArtistWidget(QWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        records: Sequence[ImageRecord],
        output_root: Path,
        rater_name: str,
        start_index: int = 0,
        include_completed: bool = False,
        write_sidecars: bool = False,
    ) -> None:
        super().__init__()
        self.viewer = viewer
        self.records = list(records)
        self.output_root = output_root
        self.rater_name = sanitize_label(rater_name)
        self.include_completed = include_completed
        self.write_sidecars = write_sidecars
        self.current_index = min(max(start_index, 0), max(len(self.records) - 1, 0))
        self.current_record: ImageRecord | None = None
        self.current_image: nib.Nifti1Image | None = None
        self.current_data: np.ndarray | None = None
        self.current_curve: np.ndarray | None = None
        self.current_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.current_window_control_range: tuple[float, float] = (0.0, 1.0)
        self.current_ornt_source_to_display: np.ndarray = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        self.current_ornt_display_to_source: np.ndarray = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        self.current_2d_order: tuple[int, int, int] = DEFAULT_2D_ORDER
        self.current_2d_slice_indices: dict[int, int] = {0: 0, 1: 0, 2: 0}
        self.has_entered_2d_view = False
        self._frame_scroll_progress = 0.0
        self.current_intensity_bounds: tuple[float, float] = (0.0, 1.0)
        self.image_layer: Image | None = None
        self.labels_layer: Labels | None = None
        self._updating_window_controls = False
        self._record_load_request_id = 0
        self._record_load_in_progress = False
        self._prefetch_request_id = 0
        self._prefetched_index: int | None = None
        self._prefetched_record: LoadedRecord | None = None
        self.save_thread_pool = QThreadPool(self)
        self.pending_save_count = 0
        self.setMinimumWidth(0)

        self._curve_timer = QTimer(self)
        self._curve_timer.setInterval(150)
        self._curve_timer.setSingleShot(True)
        self._curve_timer.timeout.connect(self._refresh_curve)
        self.viewer._aifartist_widget = self
        if aifartist_right_click_erase not in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.insert(0, aifartist_right_click_erase)
        self.viewer.mouse_wheel_callbacks.clear()
        self.viewer.mouse_wheel_callbacks.append(aifartist_mouse_wheel)
        self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)
        self.viewer.dims.events.order.connect(self._on_dims_order_changed)
        self.viewer.dims.events.current_step.connect(self._on_current_step_changed)

        self._build_ui()
        self._ensure_derivative_description()
        if not self.include_completed:
            self._jump_to_first_unsaved(start_index)
        self.load_record(self.current_index)

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        info_group = QGroupBox("Session")
        info_layout = QFormLayout()
        info_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.rater_edit = QLineEdit(self.rater_name)
        self.rater_edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.rater_edit.editingFinished.connect(self._on_rater_changed)
        self.position_label = QLabel("0 / 0")
        self.file_label = QLabel("-")
        self.file_label.setWordWrap(True)
        self.file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.roi_summary_label = QLabel("0 voxels")
        info_layout.addRow("Rater", self.rater_edit)
        info_layout.addRow("Queue", self.position_label)
        info_layout.addRow("Image", self.file_label)
        info_layout.addRow("Status", self.status_label)
        info_layout.addRow("ROI", self.roi_summary_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        frame_group = QGroupBox("Display Frame")
        frame_layout = QVBoxLayout()
        slider_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.valueChanged.connect(self.frame_slider.setValue)
        self.frame_slider.valueChanged.connect(self.frame_spinbox.setValue)
        slider_row.addWidget(self.frame_slider)
        slider_row.addWidget(self.frame_spinbox)
        frame_layout.addLayout(slider_row)
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)

        window_group = QGroupBox("Window / Level")
        window_layout = QFormLayout()
        self.window_low_spinbox = QDoubleSpinBox()
        self.window_low_spinbox.setDecimals(3)
        self.window_low_spinbox.setKeyboardTracking(False)
        self.window_low_spinbox.valueChanged.connect(self._on_window_limits_changed)
        self.window_low_slider = QSlider(Qt.Horizontal)
        self.window_low_slider.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.window_low_slider.setRange(0, 1000)
        self.window_low_slider.valueChanged.connect(self._on_window_slider_changed)
        self.window_high_spinbox = QDoubleSpinBox()
        self.window_high_spinbox.setDecimals(3)
        self.window_high_spinbox.setKeyboardTracking(False)
        self.window_high_spinbox.valueChanged.connect(self._on_window_limits_changed)
        self.window_high_slider = QSlider(Qt.Horizontal)
        self.window_high_slider.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.window_high_slider.setRange(0, 1000)
        self.window_high_slider.valueChanged.connect(self._on_window_slider_changed)
        self.window_auto_button = QPushButton("Auto")
        self.window_auto_button.clicked.connect(self._auto_window_current_frame)
        self.window_auto_button.setToolTip("Auto-set contrast from the currently displayed frame")

        low_row = QHBoxLayout()
        low_row.addWidget(self.window_low_spinbox)
        low_row.addWidget(self.window_low_slider)
        high_row = QHBoxLayout()
        high_row.addWidget(self.window_high_spinbox)
        high_row.addWidget(self.window_high_slider)
        window_layout.addRow("Low", low_row)
        window_layout.addRow("High", high_row)
        window_layout.addRow("Auto", self.window_auto_button)
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)

        self.curve_canvas = CurveCanvas()
        layout.addWidget(self.curve_canvas)
        self.show_first_normalized_checkbox = QCheckBox("Show first-point normalized graph")
        self.show_first_normalized_checkbox.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.show_first_normalized_checkbox.setToolTip("Toggle the graph normalized to the first timepoint")
        self.show_first_normalized_checkbox.toggled.connect(self._update_curve_visibility)
        layout.addWidget(self.show_first_normalized_checkbox)
        self.first_normalized_curve_canvas = CurveCanvas(
            title="ROI Signal Intensity (Normalized to First Point)",
            y_label="Relative intensity",
        )
        self.first_normalized_curve_canvas.setVisible(False)
        layout.addWidget(self.first_normalized_curve_canvas)
        self.show_second_normalized_checkbox = QCheckBox("Show second-point normalized graph")
        self.show_second_normalized_checkbox.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.show_second_normalized_checkbox.setToolTip("Toggle the graph normalized to the second timepoint")
        self.show_second_normalized_checkbox.toggled.connect(self._update_curve_visibility)
        layout.addWidget(self.show_second_normalized_checkbox)
        self.second_normalized_curve_canvas = CurveCanvas(
            title="ROI Signal Intensity (Normalized to Second Point)",
            y_label="Relative intensity",
        )
        self.second_normalized_curve_canvas.setVisible(False)
        layout.addWidget(self.second_normalized_curve_canvas)

        button_row = QGridLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setHorizontalSpacing(6)
        button_row.setVerticalSpacing(6)
        self.prev_button = QPushButton("Previous")
        self.prev_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.prev_button.clicked.connect(self.load_previous)
        self.clear_button = QPushButton("Clear ROI")
        self.clear_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.clear_button.clicked.connect(self.clear_roi)
        self.skip_button = QPushButton("Skip")
        self.skip_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.skip_button.clicked.connect(self.load_next)
        self.flag_button = QPushButton("Flag + Skip")
        self.flag_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.flag_button.clicked.connect(self.flag_current_and_advance)
        self.flag_button.setToolTip("Flag the current image and move to the next one")
        button_row.addWidget(self.prev_button, 0, 0)
        button_row.addWidget(self.clear_button, 0, 1)
        button_row.addWidget(self.skip_button, 1, 0)
        button_row.addWidget(self.flag_button, 1, 1)
        layout.addLayout(button_row)

        self.save_button = QPushButton("Save ROI + Next")
        self.save_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.save_button.clicked.connect(self.save_current_and_advance)
        self.save_button.setShortcut("Ctrl+Return")
        self.save_button.setToolTip(
            "Save the current ROI and advance to the next image (Ctrl+Enter)"
        )
        layout.addWidget(self.save_button)

        layout.addStretch(1)
        self.setLayout(layout)

    def _jump_to_first_unsaved(self, requested_index: int) -> None:
        if not self.records:
            self.current_index = 0
            return
        for index in range(max(requested_index, 0), len(self.records)):
            if not existing_mask_path_for_record(
                self.output_root,
                self.records[index],
                self.rater_name,
            ).exists():
                self.current_index = index
                return
        self.current_index = min(max(requested_index, 0), len(self.records) - 1)

    def _ensure_derivative_description(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        dataset_description = self.output_root / "dataset_description.json"
        if dataset_description.exists():
            return
        dataset_description.write_text(
            json.dumps(
                {
                    "Name": "AIFArtist manual AIF derivatives",
                    "BIDSVersion": "1.10.0",
                    "DatasetType": "derivative",
                    "GeneratedBy": [
                        {
                            "Name": "AIFArtist",
                            "Version": "0.1.0",
                            "Description": "Manual arterial input function ROI annotation with napari",
                        }
                    ],
                },
                indent=2,
            )
            + "\n"
        )

    def _on_rater_changed(self) -> None:
        new_name = sanitize_label(self.rater_edit.text())
        if not new_name:
            self.rater_edit.setText(self.rater_name)
            return
        self.rater_name = new_name
        self.rater_edit.setText(new_name)
        if self.current_record is not None:
            self._set_status("Updated rater label.")
            self._load_existing_mask()

    def _bind_label_events(self) -> None:
        assert self.labels_layer is not None
        for event_name in ("paint", "set_data", "labels_update", "data"):
            getattr(self.labels_layer.events, event_name).connect(self._queue_curve_refresh)

    def _queue_curve_refresh(self, event=None) -> None:
        self._curve_timer.start()

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _clear_prefetch_state(self) -> None:
        self._prefetch_request_id += 1
        self._prefetched_index = None
        self._prefetched_record = None

    def _set_record_loading_state(self, is_loading: bool) -> None:
        self._record_load_in_progress = is_loading
        for widget in (
            self.prev_button,
            self.clear_button,
            self.skip_button,
            self.flag_button,
            self.save_button,
            self.frame_slider,
            self.frame_spinbox,
        ):
            widget.setEnabled(not is_loading)
        if self.labels_layer is not None:
            self.labels_layer.editable = not is_loading
            if is_loading:
                self.labels_layer.mode = "pan_zoom"
            else:
                self._sync_roi_interaction_mode()

    def _update_curve_visibility(self) -> None:
        self.first_normalized_curve_canvas.setVisible(self.show_first_normalized_checkbox.isChecked())
        self.second_normalized_curve_canvas.setVisible(self.show_second_normalized_checkbox.isChecked())

    def _clear_loaded_record_state(self, message: str) -> None:
        self._set_record_loading_state(False)
        self._clear_prefetch_state()
        self.current_index = 0
        self.current_record = None
        self.current_image = None
        self.current_data = None
        self.current_curve = None
        self.position_label.setText("0 / 0")
        self.file_label.setText("-")
        self.roi_summary_label.setText("0 voxels")
        self.curve_canvas.update_curve(None)
        self.first_normalized_curve_canvas.update_curve(None)
        self.second_normalized_curve_canvas.update_curve(None)
        self.frame_slider.blockSignals(True)
        self.frame_spinbox.blockSignals(True)
        self.frame_slider.setMaximum(0)
        self.frame_spinbox.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_spinbox.setValue(0)
        self.frame_slider.blockSignals(False)
        self.frame_spinbox.blockSignals(False)
        if self.image_layer is not None:
            self.image_layer.visible = False
        if self.labels_layer is not None:
            self.labels_layer.data = np.zeros_like(self.labels_layer.data, dtype=np.uint8)
            self.labels_layer.visible = False
        self._set_status(message)

    def _apply_loaded_record(self, index: int, loaded_record: LoadedRecord) -> None:
        self._clear_prefetch_state()
        self.current_index = min(max(index, 0), len(self.records) - 1)
        record = loaded_record.record
        data = loaded_record.data
        self.current_record = record
        self.current_image = loaded_record.source_image
        self.current_data = data
        self.current_curve = None
        self.current_2d_order = DEFAULT_2D_ORDER
        self.current_2d_slice_indices = {axis: data.shape[axis] // 2 for axis in range(3)}
        self.has_entered_2d_view = False
        self.current_spacing = loaded_record.spacing
        self.current_intensity_bounds = loaded_record.intensity_bounds
        self.current_ornt_source_to_display = loaded_record.source_to_display
        self.current_ornt_display_to_source = loaded_record.display_to_source

        frame_count = data.shape[-1]
        default_frame = loaded_record.default_frame
        auto_limits = compute_auto_contrast_limits(
            data[..., default_frame],
            self.current_intensity_bounds,
        )
        preserved_limits = auto_limits
        if self.image_layer is not None:
            current_limits = tuple(float(value) for value in self.image_layer.contrast_limits)
            if len(current_limits) == 2 and np.all(np.isfinite(current_limits)) and current_limits[0] < current_limits[1]:
                preserved_limits = current_limits
        self.frame_slider.blockSignals(True)
        self.frame_spinbox.blockSignals(True)
        self.frame_slider.setMaximum(frame_count - 1)
        self.frame_spinbox.setMaximum(frame_count - 1)
        self.frame_slider.setValue(default_frame)
        self.frame_spinbox.setValue(default_frame)
        self.frame_slider.blockSignals(False)
        self.frame_spinbox.blockSignals(False)

        if self.image_layer is None:
            self.image_layer = self.viewer.add_image(
                data[..., default_frame],
                name="Dynamic 3D volume",
                colormap="gray",
                rendering="mip",
                depiction="volume",
                scale=self.current_spacing,
                contrast_limits=preserved_limits,
            )
        else:
            self.image_layer.data = data[..., default_frame]
            self.image_layer.scale = self.current_spacing
            self.image_layer.contrast_limits = preserved_limits
        self.image_layer.visible = True

        control_bounds = (
            min(self.current_intensity_bounds[0], preserved_limits[0]),
            max(self.current_intensity_bounds[1], preserved_limits[1]),
        )
        self._configure_window_level_controls(control_bounds)
        self._set_window_level_controls(*preserved_limits)

        if self.labels_layer is None:
            self.labels_layer = self.viewer.add_labels(
                np.zeros(data.shape[:3], dtype=np.uint8),
                name="AIF ROI",
                scale=self.current_spacing,
            )
            self.labels_layer.opacity = 0.55
            self.labels_layer.brush_size = 3
            self.labels_layer.n_edit_dimensions = 2
            self.labels_layer.selected_label = 1
            self._bind_label_events()
        else:
            self.labels_layer.data = np.zeros(data.shape[:3], dtype=np.uint8)
            self.labels_layer.scale = self.current_spacing
            self.labels_layer.brush_size = 3
            self.labels_layer.n_edit_dimensions = 2
            self.labels_layer.selected_label = 1
        if loaded_record.existing_mask is not None:
            self.labels_layer.data = np.asarray(loaded_record.existing_mask, dtype=np.uint8)
        self.labels_layer.visible = True

        self.viewer.dims.ndisplay = 3
        self._apply_default_view(reset_camera=True)
        self.position_label.setText(f"{self.current_index + 1} / {len(self.records)}")
        self.file_label.setText(str(record.image_path))
        self._set_status(
            f"Loaded frame {default_frame} with the largest increase in average signal intensity, "
            "coronal view, and voxel spacing "
            f"{self.current_spacing[0]:.2f} x {self.current_spacing[1]:.2f} x {self.current_spacing[2]:.2f} mm."
        )
        self._refresh_curve()
        self._request_prefetch(self.current_index + 1)

    def _request_prefetch(self, index: int) -> None:
        if not self.records or index < 0 or index >= len(self.records):
            self._clear_prefetch_state()
            return
        if self._prefetched_index == index and self._prefetched_record is not None:
            return

        self._prefetch_request_id += 1
        request_id = self._prefetch_request_id
        self._prefetched_index = None
        self._prefetched_record = None

        task = LoadTask(
            request_id,
            index,
            self.records[index],
            self.output_root,
            self.rater_name,
        )
        task.signals.finished.connect(self._on_prefetch_loaded)
        task.signals.failed.connect(self._on_prefetch_failed)
        self.save_thread_pool.start(task)

    def _on_prefetch_loaded(
        self,
        request_id: int,
        index: int,
        loaded_record: LoadedRecord,
    ) -> None:
        if request_id != self._prefetch_request_id:
            return
        self._prefetched_index = index
        self._prefetched_record = loaded_record

    def _on_prefetch_failed(
        self,
        request_id: int,
        index: int,
        record_name: str,
        error_message: str,
    ) -> None:
        if request_id != self._prefetch_request_id:
            return
        self._prefetched_index = None
        self._prefetched_record = None

    def _request_background_record_load(self, index: int, status_message: str) -> None:
        if not self.records:
            self._clear_loaded_record_state("No input images found.")
            return

        target_index = min(max(index, 0), len(self.records) - 1)
        if self._prefetched_index == target_index and self._prefetched_record is not None:
            self._apply_loaded_record(target_index, self._prefetched_record)
            self._set_record_loading_state(False)
            self._set_status(status_message)
            return

        self._record_load_request_id += 1
        request_id = self._record_load_request_id
        self._set_record_loading_state(True)
        self._set_status(status_message)

        task = LoadTask(
            request_id,
            target_index,
            self.records[target_index],
            self.output_root,
            self.rater_name,
        )
        task.signals.finished.connect(self._on_background_record_loaded)
        task.signals.failed.connect(self._on_background_record_failed)
        self.save_thread_pool.start(task)

    def _on_background_record_loaded(
        self,
        request_id: int,
        index: int,
        loaded_record: LoadedRecord,
    ) -> None:
        if request_id != self._record_load_request_id:
            return
        self._apply_loaded_record(index, loaded_record)
        self._set_record_loading_state(False)

    def _on_background_record_failed(
        self,
        request_id: int,
        index: int,
        record_name: str,
        error_message: str,
    ) -> None:
        if request_id != self._record_load_request_id:
            return
        self._set_record_loading_state(False)
        self._show_error(f"Failed to load {record_name}: {error_message}")

    def step_frame_from_scroll(self, signed_delta: float) -> None:
        if self.current_data is None:
            return

        self._frame_scroll_progress -= signed_delta
        while abs(self._frame_scroll_progress) >= 1:
            step_offset = -1 if self._frame_scroll_progress < 0 else 1
            next_frame = int(np.clip(self.frame_slider.value() + step_offset, 0, self.current_data.shape[-1] - 1))
            if next_frame == self.frame_slider.value():
                self._frame_scroll_progress = 0.0
                return
            self.frame_slider.setValue(next_frame)
            self._frame_scroll_progress -= step_offset

    def _on_frame_changed(self, frame_index: int) -> None:
        if self.current_data is None or self.image_layer is None:
            return
        clamped_index = int(np.clip(frame_index, 0, self.current_data.shape[-1] - 1))
        if clamped_index != frame_index:
            self.frame_slider.setValue(clamped_index)
            return
        self.image_layer.data = self.current_data[..., clamped_index]

    def adjust_window_limit_from_scroll(self, bound: str, signed_delta: float) -> None:
        if self.image_layer is None:
            return

        low_value, high_value = (float(value) for value in self.image_layer.contrast_limits)
        control_lower, control_upper = self.current_window_control_range
        step_size = max(self.window_high_spinbox.singleStep(), 0.001)
        minimum_gap = step_size
        delta_step = -signed_delta * step_size

        if bound == "high":
            high_value = float(np.clip(high_value + delta_step, low_value + minimum_gap, control_upper))
        elif bound == "low":
            low_value = float(np.clip(low_value + delta_step, control_lower, high_value - minimum_gap))
        else:
            raise ValueError(f"Unsupported window bound: {bound}")

        self.image_layer.contrast_limits = (low_value, high_value)
        self._set_window_level_controls(low_value, high_value)

    def _configure_window_level_controls(self, bounds: tuple[float, float]) -> None:
        lower_bound, upper_bound = bounds
        if not np.isfinite(lower_bound) or not np.isfinite(upper_bound) or lower_bound == upper_bound:
            lower_bound, upper_bound = 0.0, 1.0

        control_lower, control_upper = compute_window_control_range((lower_bound, upper_bound))
        self.current_window_control_range = (control_lower, control_upper)

        step = max((control_upper - control_lower) / 500.0, 0.001)
        self.window_low_spinbox.blockSignals(True)
        self.window_high_spinbox.blockSignals(True)
        self.window_low_spinbox.setRange(control_lower, control_upper)
        self.window_high_spinbox.setRange(control_lower, control_upper)
        self.window_low_spinbox.setSingleStep(step)
        self.window_high_spinbox.setSingleStep(step)
        self.window_low_spinbox.blockSignals(False)
        self.window_high_spinbox.blockSignals(False)

    def _set_window_level_controls(self, low_value: float, high_value: float) -> None:
        self._updating_window_controls = True
        self.window_low_spinbox.setValue(low_value)
        self.window_high_spinbox.setValue(high_value)
        self.window_low_slider.setValue(self._window_value_to_slider_position(low_value))
        self.window_high_slider.setValue(self._window_value_to_slider_position(high_value))
        self._updating_window_controls = False

    def _window_value_to_slider_position(self, value: float) -> int:
        lower_bound, upper_bound = self.current_window_control_range
        if upper_bound <= lower_bound:
            return 0
        fraction = (value - lower_bound) / (upper_bound - lower_bound)
        clamped_fraction = float(np.clip(fraction, 0.0, 1.0))
        return int(round(clamped_fraction * self.window_low_slider.maximum()))

    def _slider_position_to_window_value(self, position: int) -> float:
        lower_bound, upper_bound = self.current_window_control_range
        maximum = max(self.window_low_slider.maximum(), 1)
        fraction = position / maximum
        return lower_bound + fraction * (upper_bound - lower_bound)

    def _on_window_limits_changed(self, value: float) -> None:
        if self._updating_window_controls or self.image_layer is None:
            return

        low_value = self.window_low_spinbox.value()
        high_value = self.window_high_spinbox.value()
        if low_value >= high_value:
            if self.sender() is self.window_low_spinbox:
                low_value = min(low_value, high_value - self.window_low_spinbox.singleStep())
            else:
                high_value = max(high_value, low_value + self.window_high_spinbox.singleStep())
            self._set_window_level_controls(low_value, high_value)

        self.image_layer.contrast_limits = (low_value, high_value)

    def _on_window_slider_changed(self, position: int) -> None:
        if self._updating_window_controls:
            return

        low_value = self._slider_position_to_window_value(self.window_low_slider.value())
        high_value = self._slider_position_to_window_value(self.window_high_slider.value())
        minimum_gap = max(self.window_low_spinbox.singleStep(), 0.001)
        if low_value >= high_value:
            if self.sender() is self.window_low_slider:
                low_value = min(low_value, high_value - minimum_gap)
            else:
                high_value = max(high_value, low_value + minimum_gap)
        self._set_window_level_controls(low_value, high_value)
        if self.image_layer is not None:
            self.image_layer.contrast_limits = (low_value, high_value)

    def _auto_window_current_frame(self) -> None:
        if self.current_data is None or self.image_layer is None:
            return
        frame_index = int(self.frame_slider.value())
        auto_limits = compute_auto_contrast_limits(
            self.current_data[..., frame_index],
            self.current_intensity_bounds,
        )
        self.image_layer.contrast_limits = auto_limits
        self._set_window_level_controls(*auto_limits)

    def _on_ndisplay_changed(self, event=None) -> None:
        self._apply_default_view(reset_camera=False)

    def _on_dims_order_changed(self, event=None) -> None:
        self._store_current_2d_view_state()

    def _on_current_step_changed(self, event=None) -> None:
        self._store_current_2d_view_state()

    def _store_current_2d_view_state(self) -> None:
        if self.current_data is None or self.viewer.dims.ndisplay != 2:
            return

        self.current_2d_order = normalize_2d_order(self.viewer.dims.order)
        hidden_axis = self.current_2d_order[0]
        self.current_2d_slice_indices[hidden_axis] = int(self.viewer.dims.current_step[hidden_axis])

    def _apply_default_view(self, reset_camera: bool) -> None:
        if self.current_data is None:
            return

        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "mm"

        if self.viewer.dims.ndisplay == 2:
            if not self.has_entered_2d_view:
                self.current_2d_order = DEFAULT_2D_ORDER
            self.current_2d_order = normalize_2d_order(self.current_2d_order)
            hidden_axis = self.current_2d_order[0]
            default_slice_index = self.current_data.shape[hidden_axis] // 2
            slice_index = self.current_2d_slice_indices.get(hidden_axis, default_slice_index)

            self.viewer.dims.order = self.current_2d_order
            self.viewer.dims.set_current_step(
                hidden_axis,
                int(np.clip(slice_index, 0, self.current_data.shape[hidden_axis] - 1)),
            )
            self.viewer.camera.mouse_zoom = False
            if reset_camera:
                self.viewer.reset_view()
            self.has_entered_2d_view = True
            self._sync_roi_interaction_mode()
            return

        self.viewer.camera.mouse_zoom = True
        if reset_camera:
            self.viewer.reset_view()
            self.viewer.camera.set_view_direction((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self._sync_roi_interaction_mode()

    def _sync_roi_interaction_mode(self) -> None:
        if self.labels_layer is None:
            return

        self.labels_layer.editable = True
        self.labels_layer.visible = True
        self.viewer.layers.selection.active = self.labels_layer
        if self.viewer.dims.ndisplay == 2:
            self.labels_layer.mode = "paint"
        else:
            self.labels_layer.mode = "pan_zoom"

    def load_record(self, index: int) -> None:
        if not self.records:
            self._clear_loaded_record_state("No input images found.")
            return

        self._record_load_request_id += 1
        self._set_record_loading_state(False)
        target_index = min(max(index, 0), len(self.records) - 1)
        if self._prefetched_index == target_index and self._prefetched_record is not None:
            self._apply_loaded_record(target_index, self._prefetched_record)
            return

        record = self.records[target_index]
        try:
            loaded_record = prepare_record_for_display(record, self.output_root, self.rater_name)
        except Exception as exc:
            self._show_error(f"Failed to load {record.image_path.name}: {exc}")
            return
        self._apply_loaded_record(target_index, loaded_record)

    def _load_existing_mask(self) -> None:
        if self.labels_layer is None or self.current_record is None:
            return
        mask_path = existing_mask_path_for_record(
            self.output_root,
            self.current_record,
            self.rater_name,
        )
        if not mask_path.exists():
            self._set_status("Loaded image.")
            return
        try:
            mask = np.asarray(nib.load(str(mask_path)).get_fdata(), dtype=np.uint8)
        except Exception as exc:
            self._show_error(f"Failed to load existing ROI: {exc}")
            return
        mask = reorient_spatial_array(mask, self.current_ornt_source_to_display)
        if self.current_data is not None and mask.shape != self.current_data.shape[:3]:
            self._show_error(
                f"Existing ROI shape {mask.shape} does not match image shape {self.current_data.shape[:3]}."
            )
            return
        self.labels_layer.data = np.asarray(mask, dtype=np.uint8)
        self._set_status("Loaded existing ROI for this rater.")

    def _refresh_curve(self) -> None:
        if self.current_data is None or self.labels_layer is None:
            self.curve_canvas.update_curve(None)
            self.first_normalized_curve_canvas.update_curve(None)
            self.second_normalized_curve_canvas.update_curve(None)
            self.roi_summary_label.setText("0 voxels")
            return

        label_data = np.asarray(self.labels_layer.data)
        mask = np.asarray(label_data > 0)
        voxel_count = int(mask.sum())
        self.roi_summary_label.setText(f"{voxel_count} voxels")
        if voxel_count == 0:
            self.current_curve = None
            self.curve_canvas.update_curve(None)
            self.first_normalized_curve_canvas.update_curve(None)
            self.second_normalized_curve_canvas.update_curve(None)
            return

        roi_matrix = self.current_data[mask]
        self.current_curve = roi_matrix.mean(axis=0)
        label_curves: dict[int, np.ndarray] = {}
        label_curve_colors: dict[int, tuple[float, float, float, float]] = {}
        for label_value in sorted(int(value) for value in np.unique(label_data) if value > 0):
            label_mask = label_data == label_value
            if not np.any(label_mask):
                continue
            label_curves[label_value] = self.current_data[label_mask].mean(axis=0)
            label_color = self.labels_layer.get_color(label_value)
            if label_color is not None:
                label_curve_colors[label_value] = tuple(float(channel) for channel in label_color)
        self.curve_canvas.update_curve(label_curves, label_curve_colors)
        first_point_curves = normalize_curves_to_point(label_curves, 0)
        second_point_curves = normalize_curves_to_point(label_curves, 1)
        self.first_normalized_curve_canvas.update_curve(
            first_point_curves,
            label_curve_colors,
            empty_message="Normalization to the first point is unavailable for the current labels.",
        )
        self.second_normalized_curve_canvas.update_curve(
            second_point_curves,
            label_curve_colors,
            empty_message="Normalization to the second point is unavailable for the current labels.",
        )

    def clear_roi(self) -> None:
        if self.labels_layer is None:
            return
        self.labels_layer.data = np.zeros_like(self.labels_layer.data, dtype=np.uint8)
        self._set_status("Cleared ROI.")
        self._refresh_curve()

    def load_previous(self) -> None:
        if self.current_index <= 0:
            self._set_status("Already at the first image.")
            return
        self.load_record(self.current_index - 1)

    def load_next(self) -> None:
        if self.current_index >= len(self.records) - 1:
            self._set_status("Reached the last image.")
            return
        self.load_record(self.current_index + 1)

    def flag_current_and_advance(self) -> None:
        if self.current_record is None:
            return

        reason, accepted = QInputDialog.getItem(
            self,
            "Flag image",
            "Reason for flagging:",
            list(FLAG_REASON_OPTIONS),
            0,
            False,
        )
        if not accepted or not reason:
            return

        try:
            append_flagged_record(self.output_root, self.current_record, self.rater_name, reason)
        except Exception as exc:
            self._show_error(f"Failed to record image flag: {exc}")
            return

        flagged_name = self.current_record.image_path.name
        self.records.pop(self.current_index)
        if not self.records:
            self._clear_loaded_record_state(
                f"Flagged {flagged_name} as '{reason}'. No remaining images in the queue."
            )
            return

        next_index = min(self.current_index, len(self.records) - 1)
        self.load_record(next_index)
        self._set_status(f"Flagged {flagged_name} as '{reason}' and skipped it.")

    def save_current_and_advance(self) -> None:
        if self.current_record is None or self.current_image is None or self.labels_layer is None:
            return

        display_mask = np.asarray(self.labels_layer.data > 0, dtype=np.uint8)
        mask = reorient_spatial_array(display_mask, self.current_ornt_display_to_source)
        voxel_count = int(mask.sum())
        if voxel_count == 0:
            self._show_error("Draw at least one voxel before saving.")
            return

        curve = self.current_curve
        if curve is None:
            self._refresh_curve()
            curve = self.current_curve
        if curve is None:
            self._show_error("Could not compute the AIF curve for this ROI.")
            return

        mask_path = self.mask_path_for(self.current_record)
        tsv_path = self.timeseries_path_for(self.current_record)
        json_path = self.metadata_path_for(self.current_record)
        save_request = SaveRequest(
            record_stem=self.current_record.stem,
            image_path=str(self.current_record.image_path),
            rater_name=self.rater_name,
            image_shape=tuple(self.current_image.shape),
            mask=np.asarray(mask, dtype=np.uint8).copy(),
            affine=np.asarray(self.current_image.affine, dtype=float).copy(),
            header=self.current_image.header.copy(),
            curve=np.asarray(curve, dtype=np.float32).copy(),
            mask_path=mask_path,
            tsv_path=tsv_path,
            json_path=json_path,
            write_sidecars=self.write_sidecars,
        )
        self._queue_background_save(save_request)

        if self.current_index < len(self.records) - 1:
            next_index = self.current_index + 1
            if self._prefetched_index == next_index and self._prefetched_record is not None:
                self._apply_loaded_record(next_index, self._prefetched_record)
                self._set_status(
                    f"Queued background save for {save_request.record_stem}. "
                    f"Showing next image immediately. Pending saves: {self.pending_save_count}."
                )
            else:
                self._request_background_record_load(
                    next_index,
                    f"Queued background save for {save_request.record_stem}. Loading next image... "
                    f"Pending saves: {self.pending_save_count}.",
                )
        else:
            self._set_status(
                f"Queued final ROI save in background. Pending saves: {self.pending_save_count}."
            )

    def _queue_background_save(self, request: SaveRequest) -> None:
        task = SaveTask(request)
        task.signals.finished.connect(self._on_background_save_finished)
        task.signals.failed.connect(self._on_background_save_failed)
        self.pending_save_count += 1
        self.save_thread_pool.start(task)

    def _on_background_save_finished(self, record_stem: str) -> None:
        self.pending_save_count = max(self.pending_save_count - 1, 0)
        if self.pending_save_count == 0 and not self._record_load_in_progress:
            self._set_status("Background ROI saves complete.")

    def _on_background_save_failed(self, record_stem: str, error_message: str) -> None:
        self.pending_save_count = max(self.pending_save_count - 1, 0)
        self._show_error(f"Background save failed for {record_stem}: {error_message}")

    def closeEvent(self, event) -> None:
        if self.pending_save_count > 0:
            self._set_status("Waiting for background ROI saves to finish...")
            self.save_thread_pool.waitForDone()
        super().closeEvent(event)

    def output_directory_for(self, record: ImageRecord) -> Path:
        path = self.output_root / f"sub-{record.subject}"
        if record.session:
            path = path / f"ses-{record.session}"
        return path / "dce"

    def output_stem_for(self, record: ImageRecord) -> str:
        parts = [record.entity_stem, f"desc-rater{self.rater_name}", "label-AIF"]
        return "_".join(parts)

    def mask_path_for(self, record: ImageRecord) -> Path:
        return self.output_directory_for(record) / f"{self.output_stem_for(record)}_mask.nii.gz"

    def timeseries_path_for(self, record: ImageRecord) -> Path:
        return self.output_directory_for(record) / f"{self.output_stem_for(record)}_timeseries.tsv"

    def metadata_path_for(self, record: ImageRecord) -> Path:
        return self.output_directory_for(record) / f"{self.output_stem_for(record)}_timeseries.json"

    def _show_error(self, message: str) -> None:
        self._set_status(message)
        QMessageBox.critical(self, "AIFArtist", message)


def sanitize_label(value: str) -> str:
    cleaned = "".join(char for char in value.strip() if char.isalnum())
    return cleaned or "anon"


def infer_max_increase_frame(data: np.ndarray) -> int:
    global_curve = data.reshape(-1, data.shape[-1]).mean(axis=0)
    if global_curve.size <= 1:
        return 0

    signal_increase = np.diff(global_curve)
    return int(np.argmax(signal_increase) + 1)


def compute_intensity_bounds(data: np.ndarray) -> tuple[float, float]:
    finite_values = np.asarray(data[np.isfinite(data)], dtype=np.float32)
    if finite_values.size == 0:
        return (0.0, 1.0)

    lower_bound = float(finite_values.min())
    upper_bound = float(finite_values.max())
    if lower_bound == upper_bound:
        return (lower_bound - 0.5, upper_bound + 0.5)
    return (lower_bound, upper_bound)


def compute_auto_contrast_limits(
    frame_data: np.ndarray,
    fallback_bounds: tuple[float, float],
) -> tuple[float, float]:
    finite_values = np.asarray(frame_data[np.isfinite(frame_data)], dtype=np.float32)
    if finite_values.size == 0:
        return fallback_bounds

    low_value = float(np.percentile(finite_values, 1.0))
    high_value = float(np.percentile(finite_values, 99.5))
    if low_value >= high_value:
        return fallback_bounds
    return (low_value, high_value)


def compute_window_control_range(bounds: tuple[float, float]) -> tuple[float, float]:
    lower_bound, upper_bound = bounds
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound) or lower_bound == upper_bound:
        return (0.0, 1.0)

    span = upper_bound - lower_bound
    padding = max(span * 0.25, abs(upper_bound) * 0.1, 1.0)
    return (lower_bound - padding, upper_bound + padding)


def normalize_curves_to_point(
    curves: dict[int, np.ndarray],
    point_index: int,
) -> dict[int, np.ndarray]:
    normalized_curves: dict[int, np.ndarray] = {}
    for label_value, curve in curves.items():
        if curve.size <= point_index:
            continue
        baseline_value = float(curve[point_index])
        if not np.isfinite(baseline_value) or baseline_value == 0.0:
            continue
        normalized_curves[label_value] = np.asarray(curve / baseline_value, dtype=np.float32)
    return normalized_curves


def write_aif_outputs(request: SaveRequest) -> None:
    request.mask_path.parent.mkdir(parents=True, exist_ok=True)

    mask_header = request.header.copy()
    mask_header.set_data_dtype(np.uint8)
    mask_header.set_data_shape(request.mask.shape)
    mask_image = nib.Nifti1Image(request.mask, affine=request.affine, header=mask_header)
    nib.save(mask_image, str(request.mask_path))

    if not request.write_sidecars:
        return

    curve_table = pd.DataFrame(
        {
            "time_index": np.arange(request.curve.size, dtype=int),
            "mean_signal": request.curve,
        }
    )
    curve_table.to_csv(request.tsv_path, sep="\t", index=False)

    metadata = {
        "InputImage": request.image_path,
        "ManualRater": request.rater_name,
        "ROIVoxelCount": int(request.mask.sum()),
        "ImageShape": list(request.image_shape),
        "MaskFile": request.mask_path.name,
        "CurveFile": request.tsv_path.name,
        "Created": datetime.now(timezone.utc).isoformat(),
        "SoftwareName": "AIFArtist",
        "SoftwareVersion": "0.1.0",
    }
    request.json_path.write_text(json.dumps(metadata, indent=2) + "\n")


def extract_spatial_scale(image: nib.Nifti1Image) -> tuple[float, float, float]:
    zooms = image.header.get_zooms()[:3]
    if len(zooms) != 3:
        return (1.0, 1.0, 1.0)

    spacing: list[float] = []
    for zoom in zooms:
        numeric_zoom = float(zoom)
        spacing.append(numeric_zoom if numeric_zoom > 0 else 1.0)
    return tuple(spacing)


def reorient_spatial_array(data: np.ndarray, orientation_transform: np.ndarray) -> np.ndarray:
    return np.asarray(apply_orientation(data, orientation_transform), dtype=data.dtype)


def parse_bids_record(path: Path) -> ImageRecord:
    name = path.name
    stem = name[:-7] if name.endswith(".nii.gz") else path.stem
    entity_stem = extract_entity_stem(stem)

    subject = "unknown"
    session = None
    datatype = infer_datatype(path)

    for part in path.parts:
        if part.startswith("sub-"):
            subject = part.removeprefix("sub-")
        elif part.startswith("ses-"):
            session = part.removeprefix("ses-")

    for token in stem.split("_"):
        if token.startswith("sub-"):
            subject = token.removeprefix("sub-")
        elif token.startswith("ses-"):
            session = token.removeprefix("ses-")

    return ImageRecord(
        image_path=path,
        subject=subject,
        session=session,
        stem=stem,
        entity_stem=entity_stem,
        datatype=datatype,
    )


def extract_entity_stem(stem: str) -> str:
    tokens = stem.split("_")
    entity_tokens = [token for token in tokens if "-" in token]
    if entity_tokens:
        return "_".join(entity_tokens)
    return stem


def infer_datatype(path: Path) -> str:
    known_datatypes = {
        "anat",
        "dwi",
        "fmap",
        "func",
        "perf",
        "pet",
        "micr",
        "meg",
        "eeg",
        "ieeg",
    }
    for part in reversed(path.parts[:-1]):
        if part in known_datatypes:
            return part
    return "func"


def output_directory_for(output_root: Path, record: ImageRecord) -> Path:
    path = output_root / f"sub-{record.subject}"
    if record.session:
        path = path / f"ses-{record.session}"
    return path / "dce"


def output_stem_for(record: ImageRecord, rater_name: str) -> str:
    parts = [record.entity_stem, f"desc-rater{sanitize_label(rater_name)}", "label-AIF"]
    return "_".join(parts)


def mask_path_for_record(output_root: Path, record: ImageRecord, rater_name: str) -> Path:
    return output_directory_for(output_root, record) / f"{output_stem_for(record, rater_name)}_mask.nii.gz"


def flagged_csv_path_for(output_root: Path, rater_name: str) -> Path:
    return output_root / f"desc-rater{sanitize_label(rater_name)}_flags.csv"


def load_flagged_image_paths(output_root: Path, rater_name: str) -> set[str]:
    csv_path = flagged_csv_path_for(output_root, rater_name)
    if not csv_path.exists():
        return set()

    flagged_images: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image_path = (row.get("img") or "").strip()
            if image_path:
                flagged_images.add(image_path)
    return flagged_images


def append_flagged_record(output_root: Path, record: ImageRecord, rater_name: str, reason: str) -> None:
    csv_path = flagged_csv_path_for(output_root, rater_name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("img", "reason"))
        if not file_exists:
            writer.writeheader()
        writer.writerow({"img": str(record.image_path), "reason": reason})


def mask_path_variants_for_record(output_root: Path, record: ImageRecord, rater_name: str) -> tuple[Path, ...]:
    output_directory = output_directory_for(output_root, record)
    sanitized_rater = sanitize_label(rater_name)
    return tuple(
        output_directory
        / f"{'_'.join((record.entity_stem, f'desc-rater{sanitized_rater}', f'label-{label_variant}'))}_mask.nii.gz"
        for label_variant in MASK_LABEL_VARIANTS
    )


def existing_mask_path_for_record(output_root: Path, record: ImageRecord, rater_name: str) -> Path:
    for mask_path in mask_path_variants_for_record(output_root, record, rater_name):
        if mask_path.exists():
            return mask_path
    return mask_path_for_record(output_root, record, rater_name)


def filter_completed_records(
    records: Sequence[ImageRecord],
    output_root: Path,
    rater_name: str,
    include_completed: bool,
) -> list[ImageRecord]:
    flagged_images = load_flagged_image_paths(output_root, rater_name)
    if include_completed:
        return [record for record in records if str(record.image_path) not in flagged_images]
    return [
        record
        for record in records
        if str(record.image_path) not in flagged_images
        and not existing_mask_path_for_record(output_root, record, rater_name).exists()
    ]


def discover_nifti_files(inputs: Sequence[str], manifest: str | None) -> list[ImageRecord]:
    discovered: list[Path] = []

    if manifest:
        manifest_path = Path(manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        if manifest_path.suffix.lower() in {".csv", ".tsv"}:
            separator = "\t" if manifest_path.suffix.lower() == ".tsv" else ","
            table = pd.read_csv(manifest_path, sep=separator)
            for column in ("path", "image", "image_path", "nifti"):
                if column in table.columns:
                    discovered.extend(resolve_manifest_entry(value, manifest_path.parent) for value in table[column])
                    break
            else:
                raise ValueError(
                    "Manifest must contain one of these columns: path, image, image_path, nifti."
                )
        else:
            discovered.extend(
                resolve_manifest_entry(line.strip(), manifest_path.parent)
                for line in manifest_path.read_text().splitlines()
                if line.strip()
            )

    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        if path.is_dir():
            discovered.extend(find_nifti_files(path))
        elif is_motion_corrected_source_path(path):
            discovered.append(path)

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    records = [parse_bids_record(path) for path in unique_paths if is_valid_source_image(path)]
    records.sort(key=lambda record: str(record.image_path))
    return records


def resolve_manifest_entry(value: str, manifest_parent: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_parent / candidate
    return candidate.resolve()


def find_nifti_files(root: Path) -> list[Path]:
    results: list[Path] = []
    for suffix in SOURCE_IMAGE_SUFFIXES:
        for path in sorted(root.rglob(f"*{suffix}")):
            results.append(path)
    return results


def is_motion_corrected_source_path(path: Path) -> bool:
    return path.name.endswith(SOURCE_IMAGE_SUFFIXES)


def is_valid_source_image(path: Path) -> bool:
    if not is_motion_corrected_source_path(path):
        return False
    try:
        image = nib.load(str(path))
    except Exception:
        return False
    return len(image.shape) == 4


def infer_output_root(records: Sequence[ImageRecord], requested_output_root: str | None) -> Path:
    if requested_output_root:
        return Path(requested_output_root).expanduser().resolve()
    return DEFAULT_OUTPUT_ROOT


def ask_for_rater_name(initial_value: str | None = None) -> str:
    value, accepted = QInputDialog.getText(
        None,
        "AIFArtist",
        "Enter the rater name or initials used in output files:",
        text=initial_value or "",
    )
    if not accepted:
        raise SystemExit(1)
    return sanitize_label(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive napari desktop tool for manual AIF ROI definition on 4D NIfTI MRI data."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="One or more BIDS-compliant desc-hmc_DCE.nii[.gz] files or directories to scan recursively.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional text/CSV/TSV manifest listing BIDS-compliant desc-hmc_DCE.nii[.gz] images to annotate.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory for BIDS-style derivative outputs. Defaults near the input dataset.",
    )
    parser.add_argument(
        "--rater",
        help="Rater name or initials. If omitted, the GUI prompts for it.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Queue index to open first. By default, the app jumps to the first unsaved image.",
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Include images that already have an AIF ROI saved by the current rater.",
    )
    parser.add_argument(
        "--write-sidecars",
        action="store_true",
        help="Also write the ROI timeseries TSV and metadata JSON alongside the mask.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv)

    if not args.inputs and not args.manifest:
        parser.error("Provide at least one input path or a manifest.")

    records = discover_nifti_files(args.inputs, args.manifest)
    if not records:
        parser.error("No 4D desc-hmc_DCE.nii[.gz] files found in the provided inputs.")

    rater_name = sanitize_label(args.rater) if args.rater else ask_for_rater_name(args.rater)
    output_root = infer_output_root(records, args.output_root)
    records = filter_completed_records(records, output_root, rater_name, args.include_completed)
    if not records:
        if args.include_completed:
            parser.error(
                "No remaining 4D desc-hmc_DCE.nii[.gz] files for this rater after excluding flagged images."
            )
        parser.error(
            "No remaining 4D desc-hmc_DCE.nii[.gz] files for this rater after excluding completed and flagged images. "
            "Use --include-completed to reopen completed AIFs; flagged images remain excluded."
        )

    viewer = napari.Viewer(title="AIFArtist")
    widget = AIFArtistWidget(
        viewer=viewer,
        records=records,
        output_root=output_root,
        rater_name=rater_name,
        start_index=args.start_index,
        include_completed=args.include_completed,
        write_sidecars=args.write_sidecars,
    )
    dock_widget = viewer.window.add_dock_widget(
        widget,
        area="right",
        name="AIF Session",
        menu=viewer.window.window_menu,
    )
    dock_widget.setMinimumWidth(0)

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())