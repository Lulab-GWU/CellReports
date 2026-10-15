"""Small shared helpers used by the figure notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def repository_root(start: str | Path | None = None) -> Path:
    """Return the nearest parent containing ``pyproject.toml``."""

    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate pyproject.toml. Run from inside the repository."
    )


def project_path(*parts: str | Path, root: str | Path | None = None) -> Path:
    """Construct a path relative to the repository root."""

    base = Path(root).resolve() if root is not None else repository_root()
    return base.joinpath(*map(Path, parts))


def ensure_directory(path: str | Path) -> Path:
    """Create an output directory and return its resolved path."""

    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output


def contiguous_intervals(
    mask: Sequence[bool], *, min_frames: int = 1, max_gap_frames: int = 0
) -> list[tuple[int, int]]:
    """Return inclusive intervals from a Boolean sequence."""

    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > max_gap_frames + 1) + 1
    groups = np.split(indices, breaks)
    intervals = [(int(group[0]), int(group[-1])) for group in groups]
    return [
        interval
        for interval in intervals
        if interval[1] - interval[0] + 1 >= min_frames
    ]


def native_spatial_activity_image(
    table: pd.DataFrame,
    *,
    x_column: str = "x",
    y_column: str = "y",
    value_column: str = "mean_dff",
    sigma: float = 2.0,
    fill_offset: float = 2.0,
    normalize: bool = True,
) -> np.ndarray:
    """Build the smoothed native-coordinate activity image used in the figures."""

    required = {x_column, y_column, value_column}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing spatial columns: {sorted(missing)}")

    values = table.loc[:, [x_column, y_column, value_column]].copy()
    for column in values.columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.dropna()
    if values.empty:
        raise ValueError("The spatial table contains no finite coordinates and values.")

    values = values.groupby([x_column, y_column], as_index=False)[value_column].mean()
    x = np.rint(values[x_column].to_numpy(dtype=float)).astype(int)
    y = np.rint(values[y_column].to_numpy(dtype=float)).astype(int)
    activity = values[value_column].to_numpy(dtype=float)
    x_min, x_max = int(x.min()), int(x.max())
    y_min, y_max = int(y.min()), int(y.max())
    fill_value = float(np.nanmin(activity) - fill_offset)
    image = np.full((x_max - x_min + 2, y_max - y_min + 3), fill_value)
    image[x - x_min, y - y_min] = activity
    image = gaussian_filter(image, sigma=(sigma, sigma), mode="constant")

    if normalize:
        minimum, maximum = float(np.nanmin(image)), float(np.nanmax(image))
        image = (
            (image - minimum) / (maximum - minimum)
            if maximum > minimum
            else np.zeros_like(image)
        )
    return image


def save_figure(
    figure: Any,
    path: str | Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
) -> Path:
    """Save a Matplotlib figure with consistent publication settings."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", transparent=transparent)
    return output


__all__ = [
    "contiguous_intervals",
    "ensure_directory",
    "native_spatial_activity_image",
    "project_path",
    "repository_root",
    "save_figure",
]
