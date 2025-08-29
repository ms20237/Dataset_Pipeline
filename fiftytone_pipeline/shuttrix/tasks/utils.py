import sys
import cv2
from typing import List
import math
from collections import Counter
import numpy as np
import os
import yaml
import json
import fiftyone as fo


def del_all_datasets():
    for name in fo.list_datasets():
        print(f"🗑️ Deleting dataset: {name}")
        fo.delete_dataset(name)
    print("Done!!!")    

def parse_print_args(func):
    def wrapper(*args, **kwargs):
        print("Arguments passed to function:")
        for i, arg in enumerate(args):
            print(f"arg{i + 1}: {arg}")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        return func(*args, **kwargs)
    return wrapper


def get_area_counts(dataset):
    """
        Returns the area count for each sample using metadata.
    """
    dataset.compute_metadata()
    areas = [s.metadata.width * s.metadata.height for s in dataset]
    if not areas:
        return [], []
    
    counter = Counter(areas)
    area_vals = list(map(str, counter.keys()))   # str keys for x-axis
    counts = list(counter.values())              # counts as y-axis
    return area_vals, counts


def get_pixel_sizes(dataset, bins=10):
    """
        Returns binned sqrt(area) sizes and counts using metadata.
    """
    dataset.compute_metadata()
    pixel_sizes = [math.sqrt(s.metadata.width * s.metadata.height) for s in dataset]
    if not pixel_sizes:
        return [], []

    counts, bin_edges = np.histogram(pixel_sizes, bins=bins)
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)]

    return bin_labels, counts.tolist()


def get_sample_pixel_counts(dataset):
    """
    Returns a list of pixel counts using sample metadata.

    Args:
        dataset (fiftyone.core.dataset.Dataset): The FiftyOne dataset.

    Returns:
        List[int]: A list of pixel counts.
    """
    pixel_counts = []
    for sample in dataset:
        if sample.metadata is not None:
            pixel_counts.append(sample.metadata.width * sample.metadata.height)
    return pixel_counts

def find_split(name, dataset_dir):
    for subdir in dataset_dir.rglob("*"):
        if subdir.is_dir() and subdir.name == name:
            return subdir
    return None


def get_similarity_config(path: str, dataset_name: str):
    """
    Reads the similarity config JSON for a given dataset and returns the
    `do_sim` and `threshold` values.
    
    Args:
        path (str): The directory containing the similarity config JSON files.
        dataset_name (str): The name of the dataset (and the JSON file).
    
    Returns:
        A tuple (do_sim, threshold). `do_sim` is a boolean and `threshold` is a float.
        If the file is not found or has an invalid format, it returns (False, None).
    """
    config_path = os.path.join(path, f"{dataset_name}.json")
    
    if not os.path.exists(config_path):
        print(f"[yellow]Warning: No similarity config file found at '{config_path}'. Skipping similarity task.[/yellow]")
        return False, None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        do_sim = config.get("do_sim", "no").lower() == "yes"
        threshold = float(config.get("threshold", 0.0))
        
        return do_sim, threshold
    except (IOError, json.JSONDecodeError, ValueError) as e:
        print(f"[red]Error reading similarity config for '{dataset_name}': {e}[/red]")
        return False, None


def load_config_from_file(config_path):
    """
        Loads configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_area_nbins(view, area_field="area"):
    areas = []
    for sample in view:
        for det in sample["ground_truth"]["detections"]:
            area = getattr(det, area_field, None)
            if area is not None:
                areas.append(area)
    N = len(areas)
    if N == 0:
        return 10  # fallback default
    nbins = int(np.ceil(np.log2(N) + 1))
    return nbins


def compute_bbox_stats(view, width_field, height_field):
    widths = []
    heights = []
    for sample in view:
        for det in sample["ground_truth"]["detections"]:
            width = getattr(det, width_field, None)
            height = getattr(det, height_field, None)
            if width is not None and height is not None:
                widths.append(width)
                heights.append(height)
    if not widths or not heights:
        return 10, 10, 1.0, 1.0  # fallback defaults

    nxbins = int(np.ceil(np.log2(len(widths)) + 1))
    nybins = int(np.ceil(np.log2(len(heights)) + 1))
    xrange = max(widths)
    yrange = max(heights)
    return nxbins, nybins, xrange, yrange


def compute_iou(boxA, boxB):
    # box: [x, y, w, h] normalized
    xA1, yA1, wA, hA = boxA
    xA2, yA2 = xA1 + wA, yA1 + hA
    xB1, yB1, wB, hB = boxB
    xB2, yB2 = xB1 + wB, yB1 + hB

    # intersection
    x_left = max(xA1, xB1)
    y_top = max(yA1, yB1)
    x_right = min(xA2, xB2)
    y_bottom = min(yA2, yB2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    boxA_area = wA * hA
    boxB_area = wB * hB
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


def compute_iou_nbins(view, iou_field="iou"):
    ious = []
    for sample in view:
        for det in sample["ground_truth"]["detections"]:
            iou = getattr(det, iou_field, None)
            if iou is not None:
                ious.append(iou)
    N = len(ious)
    if N == 0:
        return 10  # fallback default
    nbins = int(np.ceil(np.log2(N) + 1))
    return nbins


def get_source_tag(sample, priority_list):
    """
    Finds the most prioritized source tag for a given sample.
    Args:
        sample: A FiftyOne sample object.
        priority_list: The list of prioritized dataset names from the JSON.
    Returns:
        str: The source tag with the highest priority, or None if no prioritized tag is found.
    """
    source_tags = [t for t in sample.tags if t in priority_list]
    if source_tags:
        # Sort by the index in the priority list to find the highest priority tag
        source_tags.sort(key=priority_list.index)
        return source_tags[0]
    return None

