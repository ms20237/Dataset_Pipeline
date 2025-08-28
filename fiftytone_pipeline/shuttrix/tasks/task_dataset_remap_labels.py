import numpy as np
import argparse
import json
from tqdm import tqdm
from rich import print

import fiftyone as fo
from fiftyone import ViewField as F

from shuttrix.tasks.utils import parse_print_args


def init():
    """
        This function initializes the task for filtering a FiftyOne dataset.
        It sets up command-line arguments for the dataset name and the path to a JSON file
        containing label remapping information.
        
        Args:
            ds_name (str): Name of the FiftyOne dataset to analyze.
            json_path (str): Path to the JSON file containing label remapping.
    """
    parser = argparse.ArgumentParser(description="Analyze FiftyOne dataset.")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="dataset name exp:ds1, ds2, ...")
    
    parser.add_argument('--json_path',
                        type=str,
                        required=True,
                        help="Path to the JSON file containing label remapping.")
    
    parser.add_argument("--new_dataset",
                        action='store_true',
                        default=False,
                        help="create a new dataset for output of remapping")
    
    parser.add_argument("--remove_unmapped_labels",
                        action='store_true',
                        default=False,
                        help="If set, removes labels from the dataset that are not found in the JSON remapping file.")
    
    args = parser.parse_args()
    return args 


@parse_print_args
def run(ds_name, json_path: str, new_dataset: bool, remove_unmapped_labels: bool):
    """
    Analyzes a FiftyOne dataset by loading it and printing distinct labels.
    Remaps labels if they are in the specified remap dictionary.
    Optionally removes labels from the dataset that are not in the JSON remapping file.
    Sample metadata is computed to ensure all samples have metadata.

    Args:
        ds_name (str): Name of the FiftyOne dataset to analyze.
        json_path (str): Path to the JSON file containing label remapping.
        new_dataset (bool): If True, creates a new dataset for the remapped output.
        remove_unmapped_labels (bool): If True, removes labels not in the JSON remapping.
    """
    print(f"""
        =======================================================================================================

                                        Remapping Labels in {ds_name}

        =======================================================================================================
    """)
    # load dataset
    dataset = fo.load_dataset(ds_name)

    # Load the mapping
    with open(json_path, "r") as f:
        raw_map = json.load(f)

    # Reverse the mapping: map each old label to its new label
    label_map = {}
    json_defined_old_labels = set() # This set contains all original labels that are *mentioned* in the JSON for remapping
    for new_label, old_labels in raw_map.items():
        for old_label in old_labels:
            label_map[old_label] = new_label
            json_defined_old_labels.add(old_label)

    print("\n--- Proposed Label Remapping ---")
    if label_map:
        for old_label, new_label in label_map.items():
            print(f'"{old_label}" -> "{new_label}"')
    else:
        print("No remapping rules defined in the JSON file.")

    # Get distinct labels from the dataset before any modification
    dataset_labels_before = set(dataset.distinct("ground_truth.detections.label"))
    print(f"\n--- Current Dataset Labels ---")
    print(f"Distinct labels in the dataset BEFORE modification: {dataset_labels_before}")

    labels_to_remove_preview = set()
    if remove_unmapped_labels:
        # These are labels present in the dataset but NOT listed as an old_label in the JSON
        labels_to_remove_preview = dataset_labels_before - json_defined_old_labels
        if labels_to_remove_preview:
            print(f"\n--- Labels to be Removed ---")
            print(f"[INFO] The following labels will be REMOVED from the dataset because they are NOT explicitly defined as an 'old_label' in the JSON remapping file: {labels_to_remove_preview}")
        else:
            print("\n[INFO] All existing dataset labels are covered by the JSON remapping rules. No labels will be removed based on --remove_unmapped_labels.")
    else:
        print("\n[INFO] --remove_unmapped_labels is FALSE. Labels not defined in the JSON will be kept as they are.")


    # Confirm
    print("[red]\n--- Confirmation ---")
    choice = input("[WARNING] Do you want to apply these remapping and/or removal changes? [y/n]: ").strip().lower()

    if choice in ['y', 'yes', '1', 'true', 't']:
        print(f"\nTotal samples in the dataset BEFORE modification: {len(dataset)}")

        # Make a Copy Before Modifying
        if new_dataset:
            remapped_ds_name = f"{ds_name}_remapped"
            if fo.dataset_exists(remapped_ds_name):
                print(f"Deleting existing dataset '{remapped_ds_name}'...")
                fo.delete_dataset(remapped_ds_name)
            
            # Clone the dataset to work on a copy
            print(f"Cloning dataset '{ds_name}' to '{remapped_ds_name}'...")
            dataset = dataset.clone(remapped_ds_name)
            print(f"Dataset cloned and loaded as '{remapped_ds_name}'.")
        else:
            print("Modifying the existing dataset in-place.")

        # Apply the mapping and/or removal
        for sample in tqdm(dataset, desc="Processing labels"):
            if sample.get_field("ground_truth") and sample.ground_truth.detections:
                updated_detections = []
                for det in sample.ground_truth.detections:
                    if det.label in label_map:
                        # Label is in the map, so remap it and keep
                        det.label = label_map[det.label]
                        updated_detections.append(det)
                    elif remove_unmapped_labels:
                        # We only reach here if det.label is NOT in label_map.
                        # If remove_unmapped_labels is True, then we remove it.
                        pass # Do not append, effectively removing it
                    else:
                        # If det.label is NOT in label_map AND remove_unmapped_labels is FALSE, keep it as is.
                        updated_detections.append(det)

                sample.ground_truth.detections = updated_detections
                sample.save() # Save changes to the sample

        print(f"\nDistinct labels in the dataset AFTER modification: {dataset.distinct('ground_truth.detections.label')}")
        print(f"Total samples in the dataset: {len(dataset)}")
        print(f"Dataset remapping and/or label removal completed successfully.")

    elif choice in ['n', 'no', '0', 'false', 'f']:
        print("Operation aborted by user.")

    else:
        print("Invalid choice. Please enter 'y' or 'n'.")
    
    
if __name__ == "__main__":
    args = init()
    run(args.ds_name, args.json_path, args.new_dataset, args.remove_unmapped_labels)
    

   