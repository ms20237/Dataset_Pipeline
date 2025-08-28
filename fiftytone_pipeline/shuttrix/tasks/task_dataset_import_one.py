import os
import argparse 
import tempfile
import yaml
import json
from pathlib import Path
from rich import print

import fiftyone as fo
import fiftyone.types as fot

from shuttrix.tasks.utils import parse_print_args
from fiftyone.utils.yolo import YOLOv5DatasetImporter
from fiftyone.utils.coco import COCODetectionDatasetImporter
from fiftyone.utils.voc import VOCDetectionDatasetImporter
from fiftyone.utils.kitti import KITTIDetectionDatasetImporter


def init():
    """
        This task is for importing data to fiftyone.
        Args:
            dataset name that we want in fiftyone: --ds_name 
            dataset path: --ds_path
            dataset format: --format
            dataset splits tags in fiftyone: --tag_splits
            overwrite existing dataset: --overwrite
            
    """
    parser = argparse.ArgumentParser(description="Import datasets into a FiftyOne dataset with tags.")
    
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="dataset name exp:ds1, ds2, ...")
    
    parser.add_argument('--ds_path',
                        type=str,
                        required=True,
                        help="main dataset path that want to import in fiftyone")
    
    parser.add_argument("--format",
                        type=str,
                        choices=["yolov5", "yolov8", "coco", "voc", "kitti", "fiftyone_image_classification"],
                        required=True, 
                        help="format of your dataset")
    
    parser.add_argument('--tag_splits',
                        action='store_true', 
                        help="If provided, tags samples with their respective splits (e.g., 'train', 'val', 'test'). Defaults to False if not provided.")
    
    parser.add_argument('--overwrite',
                        action='store_true', 
                        help="If provided, deletes the FiftyOne dataset if it already exists before importing. Defaults to False if not provided.")

    args = parser.parse_args()
    return args


@parse_print_args
def run(ds_name: str, ds_path: str, format: str, tag_splits: bool=False, overwrite: bool=False):
    print(
        f"""
        =======================================================================================================
                                                
                                        IMPORT {ds_name} IN FIFTYONE
                                                
        =======================================================================================================
        """
    )
    """
        Imports dataset to FiftyOne based on parsed arguments.

        Args:
            args (argparse.Namespace): Arguments parsed by `init()`
    """
    # Overwrite an existing dataset if requested
    if overwrite and fo.dataset_exists(ds_name):
        print(f"Overwriting existing dataset '{ds_name}'...")
        fo.delete_dataset(ds_name)

    # Map the format string to the corresponding FiftyOne importer class
    importer_map = {
        "yolov5": YOLOv5DatasetImporter,
        "yolov8": YOLOv5DatasetImporter,
        "coco": COCODetectionDatasetImporter,
        "voc": VOCDetectionDatasetImporter,
        "kitti": KITTIDetectionDatasetImporter,
    }
    
    importer_cls = importer_map.get(format)

    if format == "fiftyone_image_classification":
        dataset_type = fot.FiftyOneImageClassificationDataset
    elif importer_cls:
        dataset_type = importer_cls
    else:
        print(f"Error: Unsupported format '{format}'. Please choose from {list(importer_map.keys()) + ['fiftyone_image_classification']}.")
        return

    # Find the data.yaml file if format is YOLOv5/v8
    yaml_path = None
    if format in ["yolov5", "yolov8"]:
        yaml_path = os.path.join(ds_path, "data.yaml")
        if not os.path.exists(yaml_path):
            print(f"Warning: data.yaml not found at '{yaml_path}'. Importer may fail without class list.")
            yaml_path = None

    # Check for the existence of common split directories
    splits = ["train", "val", "test"]
    split_paths = {split: os.path.join(ds_path, split) for split in splits}
    splits_exist = any(os.path.isdir(p) for p in split_paths.values())

    # Import the dataset based on the tag_splits argument
    if tag_splits and splits_exist:
        print(f"Importing dataset '{ds_name}' with splits from '{ds_path}'...")
        
        # Create an empty dataset first
        dataset = fo.Dataset(name=ds_name)

        # Iterate through splits and add them to the dataset with tags
        for split in splits:
            # Check if the split directory exists
            split_dir = os.path.join(ds_path, split)
            if os.path.isdir(split_dir):
                print(f"  - Importing '{split}' split...")
                
                # Special handling for YOLOv5/YOLOv8 to instantiate the importer
                if format in ["yolov5", "yolov8"]:
                    importer = YOLOv5DatasetImporter(
                        dataset_dir=ds_path,
                        yaml_path=yaml_path,
                        split=split 
                    )
                    dataset.add_importer(importer, tags=split)
                else:
                    dataset.add_dir(
                        dataset_dir=split_dir,
                        dataset_type=dataset_type,
                        tags=split,
                    )

    else:
        # Import the entire dataset at once if no splits are found or if tag_splits is False
        if not tag_splits:
            print(f"Importing entire dataset '{ds_name}' from '{ds_path}' without split tags...")
        else: # splits_exist is False
            print(f"No splits (train/val/test) found. Importing entire dataset '{ds_name}'...")
        
        # Special handling for YOLOv5/YOLOv8 to instantiate the importer
        if format in ["yolov5", "yolov8"]:
            importer = YOLOv5DatasetImporter(
                dataset_dir=ds_path,
                yaml_path=yaml_path,
            )
            dataset = fo.Dataset.from_importer(
                importer,
                name=ds_name
            )
        else:
            dataset = fo.Dataset.from_dir(
                dataset_dir=ds_path,
                dataset_type=dataset_type,
                name=ds_name
            )
    print(f"[green]Successfully imported {dataset.name}[]")
    print("\nDataset stats:")
    print(dataset.stats())
    
    
if __name__ == "__main__":
    args = init()
    run(**vars(args))


