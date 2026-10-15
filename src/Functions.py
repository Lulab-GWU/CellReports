"""Analysis helpers used by the example notebooks."""

from __future__ import annotations

import os
from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from oasis.functions import deconvolve
from scipy.signal import find_peaks
from scipy.stats import pearsonr


def check_frame_interval(df, which_status_to_check_frame, ifKeep=False):
    """Select frames matching a behavioral status."""

    status_codes = {
        "center": "11",
        "corner": "12",
        "sniffing": "30",
        "grooming": "31",
    }
    if which_status_to_check_frame not in status_codes:
        raise ValueError(
            "Not approved status. Choose from: center, corner, sniffing, grooming"
        )

    status_code = status_codes[which_status_to_check_frame]
    status = []
    for frame_label in df["Frame_Label"]:
        label = str(frame_label)
        label_parts = [label[i : i + 2] for i in range(0, len(label), 2)]
        status.append(int(status_code in label_parts))

    df_status = df.copy()
    df_status["status"] = status

    if ifKeep:
        return df_status.drop("Frame_Label", axis=1)

    return df_status.loc[df_status["status"] == 1].drop(
        ["status", "Frame_Label"], axis=1
    )


def event_rate(df, fps, prominence=5, width=3):
    """Calculate event rate and mean event amplitude across neurons."""

    frames, n_neuron = df.shape
    values = df.to_numpy()
    peak_sum = 0
    amplitudes = []
    amplitudes_with_zero = []

    for neuron_index in range(n_neuron):
        peaks, _ = find_peaks(
            values[:, neuron_index], prominence=prominence, width=width
        )
        peak_sum += len(peaks)
        if len(peaks):
            amplitude = values[:, neuron_index][peaks].mean()
            amplitudes.append(amplitude)
            amplitudes_with_zero.append(amplitude)
        else:
            amplitudes_with_zero.append(0)

    rate = peak_sum / (frames / fps) / n_neuron if frames and n_neuron else np.nan
    amplitude = np.mean(amplitudes)
    amplitude_with_zero = np.mean(amplitudes_with_zero)
    return rate, amplitude, amplitude_with_zero


def corr_list(df):
    """Return Pearson correlations for all unique neuron pairs."""

    correlations = []
    if df.shape[0] > 2:
        for first in range(df.shape[1]):
            for second in range(first + 1, df.shape[1]):
                correlation, _ = pearsonr(df.iloc[:, first], df.iloc[:, second])
                correlations.append(correlation)
    return correlations


def randomShuffle(
    randomShuffle_N, neuronIndex, nonfocusB_df, mouse_df, focusB_length
):
    """Build a null distribution from randomly sampled non-focus frames."""

    distribution = []
    for _ in range(randomShuffle_N):
        source = nonfocusB_df if focusB_length <= len(nonfocusB_df) else mouse_df
        distribution.append(source.iloc[:, neuronIndex].sample(focusB_length).sum())
    return distribution


def get_chunksInDF(keep_combinedChunks, df_frame_label):
    """Concatenate dataframe slices corresponding to behavioral intervals."""

    chunks = [df_frame_label.iloc[start:end] for start, end in keep_combinedChunks]
    return pd.concat(chunks)


def if_spike_exist(values, threshold):
    """Return whether any value exceeds the threshold."""

    return any(value > threshold for value in values)


def if_neuronFire_moreThanChance(
    dfWhole, neuronIndex, composite_interval_list, Fpercentage, SBpercentage
):
    """Check whether a neuron fires in the required fraction of intervals."""

    trace = dfWhole.iloc[:, neuronIndex].dropna().to_numpy()
    calcium, _, baseline, _, _ = deconvolve(trace, penalty=1)
    denoised = baseline + calcium
    spike_threshold = np.percentile(denoised, SBpercentage)

    intervals_with_spikes = 0
    for start, end in composite_interval_list:
        if if_spike_exist(denoised[start:end], spike_threshold):
            intervals_with_spikes += 1

    return intervals_with_spikes / len(composite_interval_list) >= Fpercentage


def find_compositeIntervals(
    df,
    which_status_to_check,
    intervalLength=20,
    gapMax=10,
    if_CombineAdjacentChunks=True,
):
    """Find, optionally merge, and filter intervals for a behavioral status."""

    df_status = check_frame_interval(df, which_status_to_check, ifKeep=True)
    status_indices = [
        index for index, status in enumerate(df_status["status"]) if status == 1
    ]

    chunks = []
    try:
        for _, group in groupby(
            enumerate(status_indices), lambda indexed: indexed[0] - indexed[1]
        ):
            chunk = list(map(itemgetter(1), group))
            chunks.append([chunk[0], chunk[-1]])

        if if_CombineAdjacentChunks:
            combined_chunks = [chunks[0]]
            combined_index = 0
            for chunk in chunks[1:]:
                previous = combined_chunks[combined_index]
                if chunk[0] - previous[1] <= gapMax:
                    combined_chunks[combined_index] = [previous[0], chunk[-1]]
                else:
                    combined_chunks.append(chunk)
                    combined_index += 1
        else:
            combined_chunks = chunks

        kept_chunks = [
            chunk
            for chunk in combined_chunks
            if chunk[1] - chunk[0] >= intervalLength
        ]
    except Exception as error:
        kept_chunks = chunks
        print("  ", error)

    return kept_chunks


def NeuronTrace_distribution(
    df_frame_label,
    df_center,
    df_corner,
    nonF_distribution,
    behaviorNC,
    thresholdP,
    savePath,
    neuronName,
    neuronIndex,
):
    """Plot a neuron's trace and shuffled null distribution."""

    frames = df_frame_label.index.to_list()
    center_frames = df_center.loc[df_center["status"] == 1].index.to_list()
    corner_frames = df_corner.loc[df_corner["status"] == 1].index.to_list()
    trace = df_frame_label.iloc[:, neuronIndex].to_list()

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(50, 6))
    grid = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

    axes[0] = plt.subplot(grid[0])
    axes[0].plot(frames, trace, lw=4, color="black")
    for index, frame in enumerate(center_frames):
        axes[0].axvline(
            x=frame,
            color="#fadf93",
            alpha=0.4,
            label="Center" if index == 0 else None,
        )
    for index, frame in enumerate(corner_frames):
        axes[0].axvline(
            x=frame,
            color="#b5d4a7",
            alpha=0.4,
            label="Corner" if index == 0 else None,
        )
    axes[0].legend(frameon=False)
    for spine in ("right", "top", "bottom", "left"):
        axes[0].spines[spine].set_visible(False)
    axes[0].axes.get_xaxis().set_visible(False)
    axes[0].axes.get_yaxis().set_visible(False)

    axes[1] = plt.subplot(grid[1])
    axes[1].hist(nonF_distribution, bins=100, label="nonF_distribution")
    axes[1].axvline(
        behaviorNC, color="red", lw=2, label="summed focus activity"
    )
    axes[1].legend()
    axes[1].axvline(np.percentile(nonF_distribution, 100 - thresholdP), color="grey")
    axes[1].axvline(np.percentile(nonF_distribution, thresholdP), color="grey")

    figure.suptitle(neuronName, fontsize=28, fontweight="bold")
    figure.savefig(os.path.join(savePath, neuronName + "_distribution.png"))
    plt.close(figure)


def classify_ONOFF(
    df_frame_label,
    focusB_df,
    whichStatus,
    fps,
    randomShuffle_N,
    Fpercentage,
    SBpercentage,
    thresholdP,
    savePath,
    checkChance,
    composite_interval_list,
    FBTimeLenghTresh=4,
    plotD=False,
):
    """Classify neurons as ON, OFF, or other for one behavioral status."""

    df_whole = df_frame_label.copy()
    df_center = check_frame_interval(df_frame_label, "center", ifKeep=True)
    df_corner = check_frame_interval(df_frame_label, "corner", ifKeep=True)

    focus_indices = focusB_df.index.to_list()
    nonfocus_df = df_whole.loc[~df_whole.index.isin(focus_indices)]
    _, n_neuron = df_whole.iloc[:, :-1].shape
    neuron_names = df_whole.iloc[:, :-1].columns.to_list()
    neuron_types = []

    if len(focusB_df) / fps >= FBTimeLenghTresh:
        for neuron_index in range(n_neuron):
            behavior_activity = focusB_df.iloc[:, neuron_index].sum()
            null_distribution = randomShuffle(
                randomShuffle_N,
                neuron_index,
                nonfocus_df,
                df_whole,
                len(focus_indices),
            )

            passes_activity_check = not checkChance or if_neuronFire_moreThanChance(
                df_whole,
                neuron_index,
                composite_interval_list,
                Fpercentage,
                SBpercentage,
            )
            if not passes_activity_check:
                neuron_type = "Other"
            elif behavior_activity >= np.percentile(
                null_distribution, 100 - thresholdP
            ):
                neuron_type = "ON"
            elif behavior_activity <= np.percentile(null_distribution, thresholdP):
                neuron_type = "OFF"
            else:
                neuron_type = "Other"
            neuron_types.append(neuron_type)

            if plotD:
                NeuronTrace_distribution(
                    df_frame_label,
                    df_center,
                    df_corner,
                    null_distribution,
                    behavior_activity,
                    thresholdP,
                    savePath,
                    neuron_names[neuron_index]
                    + "_"
                    + whichStatus
                    + "_"
                    + neuron_type,
                    neuron_index,
                )

    on_neurons = [
        neuron_names[index]
        for index, neuron_type in enumerate(neuron_types)
        if neuron_type == "ON"
    ]
    off_neurons = [
        neuron_names[index]
        for index, neuron_type in enumerate(neuron_types)
        if neuron_type == "OFF"
    ]
    return on_neurons, off_neurons, neuron_names


def generate_ONOFF_Summary(
    all_neuron_list,
    all_ONOFF_neuron_list,
    columns_name,
    mouse_summary,
    root_path,
    fileName,
):
    """Create and save a binary ON/OFF membership table."""

    neurons = [
        neuron
        for neuron in all_neuron_list
        if neuron not in {"Frame_Label", "status"}
    ]
    labels = [
        [int(neuron in group) for group in all_ONOFF_neuron_list]
        for neuron in neurons
    ]
    summary = pd.DataFrame(labels, index=neurons, columns=columns_name)
    summary.to_csv(os.path.join(root_path, mouse_summary, fileName))
    return summary


__all__ = [
    "NeuronTrace_distribution",
    "check_frame_interval",
    "classify_ONOFF",
    "corr_list",
    "event_rate",
    "find_compositeIntervals",
    "generate_ONOFF_Summary",
    "get_chunksInDF",
    "if_neuronFire_moreThanChance",
    "if_spike_exist",
    "randomShuffle",
]
