import os
import argparse 
import tempfile
import yaml
import json
from pathlib import Path
from rich import print

import fiftyone as fo
import fiftyone.types as fot
from fiftyone.utils.yolo import YOLOv5DatasetImporter

from shuttrix.tasks.utils import parse_print_args


def init():
    """
        This task is for importing data to fiftyone.
        Args:
            dataset name that we want in fiftyone: --ds_name 
            dataset path: --ds_dir
            dataset format: --format
            dataset splits tags in fiftyone: --tag_splits
            overwrite existing dataset: --overwrite
            
    """
    parser = argparse.ArgumentParser(description="Import datasets into a FiftyOne dataset with tags.")
    
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="dataset name exp:ds1, ds2, ...")
    
    parser.add_argument('--ds_dir',
                        type=str,
                        required=True,
                        help="main dataset path that want to import in fiftyone")
    
    parser.add_argument("--format",
                        type=str,
                        choices=["yolov5", "yolov8", "coco", "voc", "kitti", "fiftyone_image_classification"],
                        required=True, # Make format required since it's crucial
                        help="format of your dataset")
    
    parser.add_argument('--tag_splits',
                        action='store_true', # If --tag_splits is present, args.tag_splits will be True
                        help="If provided, tags samples with their respective splits (e.g., 'train', 'val', 'test'). Defaults to False if not provided.")
    
    parser.add_argument('--overwrite',
                        action='store_true', # If --overwrite is present, args.overwrite will be True
                        help="If provided, deletes the FiftyOne dataset if it already exists before importing. Defaults to False if not provided.")
    
    parser.set_defaults(overwrite=True) # This makes it True by default if --overwrite is not explicitly set to false (via --no-overwrite or similar)

    parser.add_argument('--no-overwrite',
                        dest='overwrite',
                        action='store_false',
                        help="If provided, prevents overwriting an existing dataset. Overrides --overwrite.")

    args = parser.parse_args()
    return args
    

@parse_print_args
def run(ds_name: str, ds_dir: str, dataset_format: str, tag_splits: bool, overwrite: bool):
    print(
        """
        =======================================================================================================
                                                
                                                IMPORT SINGLE DATASET IN FIFTYONE
                                                
        =======================================================================================================
        """
    )

    """
    Imports a dataset (or multiple datasets) of a specified format into FiftyOne.

    Args:
        ds_name (str): The name of the dataset to be created/loaded in FiftyOne.
        ds_dir (str): The path to the directory containing the dataset(s).
        dataset_format (str): The format of the dataset (e.g., "yolov5", "coco").
        tag_splits (bool): If True, tags samples with their respective splits (train, val, test).
        overwrite (bool): If True, deletes the FiftyOne dataset if it already exists before importing.
    """

    # Map user-friendly format names to FiftyOne's DatasetType classes
    format_map = {
        "yolov5": fo.types.YOLOv5Dataset,
        "yolov8": fo.types.YOLOv5Dataset,  
        "coco": fo.types.COCODetectionDataset,
        "fiftyone_image_classification": fo.types.FiftyOneImageClassificationDataset,
        "voc": fo.types.VOCDetectionDataset,
        "kitti": fo.types.KITTIDetectionDataset,
    }

    if dataset_format not in format_map:
        raise ValueError(f"Unsupported dataset format: {dataset_format}. "
                         f"Supported formats are: {list(format_map.keys())}")

    fiftyone_dataset_type = format_map[dataset_format]

    if fo.dataset_exists(ds_name) and overwrite:
        print(f"Dataset '{ds_name}' already exists. Deleting and re-importing as 'overwrite' is True.")
        fo.delete_dataset(ds_name)
    elif fo.dataset_exists(ds_name) and not overwrite:
        print(f"Dataset '{ds_name}' already exists. Skipping import as 'overwrite' is False.")
        return

    dataset = fo.Dataset(ds_name)
    dataset.persistent = True

    print(f"Attempting to import dataset(s) of format '{dataset_format}' from: {ds_dir}")

    # Logic to handle different dataset structures and formats
    if dataset_format in ["yolov5", "yolov8"]:
        # Check if ds_dir contains train/val/test directly (single dataset)
        is_single_yolo_dataset = any(os.path.isdir(os.path.join(ds_dir, split)) for split in ["train", "val", "test"])
        
        if is_single_yolo_dataset:
            print(f"  Importing single {dataset_format} dataset from: {ds_dir}")
            for split in ["train", "val", "test"]:
                split_dir = os.path.join(ds_dir, split)
                if os.path.isdir(split_dir):
                    images_dir = os.path.join(split_dir, "images")
                    labels_dir = os.path.join(split_dir, "labels")
                    data_yaml_path = os.path.join(ds_dir, "dataset.yaml")

                    if os.path.exists(images_dir) and os.path.exists(labels_dir) and os.path.exists(data_yaml_path):
                        print(f"    Importing '{split}' split...")
                        fiftyone_split_dataset = fo.Dataset.from_dir(
                            dataset_dir=ds_dir,
                            data_path=images_dir,
                            labels_path=labels_dir,
                            dataset_type=fiftyone_dataset_type,
                            split=split,
                        )
                        if tag_splits:
                            fiftyone_split_dataset.tag_samples(split)
                        dataset.merge_samples(fiftyone_split_dataset)
                    else:
                        print(f"    Skipping '{split}' split: Missing images, labels, or dataset.yaml in {split_dir}")
                else:
                    print(f"    Skipping '{split}' split: Directory not found at {split_dir}")
        else:
            # Assume ds_dir contains multiple YOLO sub-datasets
            print(f"  Importing multiple {dataset_format} datasets from subdirectories within: {ds_dir}")
            for sub_dir_name in os.listdir(ds_dir):
                sub_dataset_path = os.path.join(ds_dir, sub_dir_name)
                if os.path.isdir(sub_dataset_path):
                    print(f"    Processing sub-dataset: {sub_dir_name}")
                    
                    # Check if this subdirectory is a valid YOLO dataset structure
                    has_yolo_structure = False
                    for split_check in ["train", "val", "test"]:
                        if os.path.isdir(os.path.join(sub_dataset_path, split_check, "images")) and \
                           os.path.isdir(os.path.join(sub_dataset_path, split_check, "labels")) and \
                           os.path.exists(os.path.join(sub_dataset_path, "dataset.yaml")):
                            has_yolo_structure = True
                            break
                    
                    if not has_yolo_structure:
                        print(f"      Skipping '{sub_dir_name}': Not a valid {dataset_format} dataset structure.")
                        continue

                    for split in ["train", "val", "test"]:
                        split_dir = os.path.join(sub_dataset_path, split)
                        if os.path.isdir(split_dir):
                            images_dir = os.path.join(split_dir, "images")
                            labels_dir = os.path.join(split_dir, "labels")
                            data_yaml_path = os.path.join(sub_dataset_path, "dataset.yaml")

                            if os.path.exists(images_dir) and os.path.exists(labels_dir) and os.path.exists(data_yaml_path):
                                print(f"        Importing '{split}' split for '{sub_dir_name}'...")
                                fiftyone_split_dataset = fo.Dataset.from_dir(
                                    dataset_dir=sub_dataset_path,
                                    data_path=images_dir,
                                    labels_path=labels_dir,
                                    dataset_type=fiftyone_dataset_type,
                                    split=split,
                                )
                                if tag_splits:
                                    fiftyone_split_dataset.tag_samples(split)
                                dataset.merge_samples(fiftyone_split_dataset)
                            else:
                                print(f"        Skipping '{split}' split for '{sub_dir_name}': Missing images, labels, or dataset.yaml.")
                        else:
                            print(f"        Skipping '{split}' split for '{sub_dir_name}': Directory not found.")
                else:
                    print(f"    Skipping '{sub_dir_name}': Not a directory.")

    elif dataset_format == "coco":
        images_path = os.path.join(ds_dir, "images")
        annotations_path = os.path.join(ds_dir, "annotations", "instances_default.json") 
        if not os.path.exists(annotations_path): # Try common COCO split names
            annotations_path = os.path.join(ds_dir, "annotations", "instances_train2017.json")
        if not os.path.exists(annotations_path):
            annotations_path = os.path.join(ds_dir, "annotations", "instances_val2017.json")
        if not os.path.exists(annotations_path):
            annotations_path = os.path.join(ds_dir, "annotations", "instances_test2017.json")
        if os.path.exists(images_path) and os.path.exists(annotations_path):
            print(f"  Importing COCO dataset from: {ds_dir}")
            try:
                coco_dataset = fo.Dataset.from_dir(
                    dataset_type=fiftyone_dataset_type,
                    data_path=images_path,
                    labels_path=annotations_path,
                    name=f"{ds_name}_coco_import"
                )
                dataset.merge_samples(coco_dataset)
                print(f"  Successfully imported COCO data from {images_path} and {annotations_path}")
            except Exception as e:
                print(f"  Error importing COCO dataset: {e}")
                print(f"  Please ensure your COCO dataset has images in '{images_path}' and "
                      f"annotations in '{annotations_path}' (or adjust paths).")
        else:
            print(f"  Could not find expected COCO structure in {ds_dir}.")
            print(f"  Expected image directory: {images_path}")
            print(f"  Expected annotation file (tried common names): {annotations_path}")
            print(f"  Please adjust --ds_dir or modify the import function for your COCO structure.")

    # Generic import for other formats if they follow a simple `from_dir` pattern
    else:
        print(f"  Attempting to import {dataset_format} dataset using default from_dir method.")
        try:
            imported_dataset = fo.Dataset.from_dir(
                dataset_dir=ds_dir,
                dataset_type=fiftyone_dataset_type,
                name=f"{ds_name}_temp_import"
            )
            dataset.merge_samples(imported_dataset)
            print(f"  Successfully imported {dataset_format} dataset from {ds_dir}.")
        except Exception as e:
            print(f"  Error importing {dataset_format} dataset: {e}")
            print(f"  Please ensure your dataset directory {ds_dir} matches the expected structure for {dataset_format}.")


    if len(dataset) > 0:
        print(f"Successfully imported {len(dataset)} samples into FiftyOne dataset '{ds_name}'.")
        # print(f"Dataset summary:\n{dataset.head()}")
    else:
        print(f"No samples were imported. Please check your --ds_dir path and dataset structure for format '{dataset_format}'.")


if __name__ == "__main__":
    args = init()
    run(args.ds_name, args.ds_dir, args.format, args.tag_splits, args.overwrite)