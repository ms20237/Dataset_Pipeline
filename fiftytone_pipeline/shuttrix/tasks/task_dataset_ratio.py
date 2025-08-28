import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from rich import print

import fiftyone as fo

from shuttrix.tasks.utils import parse_print_args


def init():
    """
        Initializes the task for comparing dataset similarity.
        Sets up command-line arguments for the dataset name and the path to a JSON file
        containing label remapping information.
        
        Args:
            ds_name (str): Name of the FiftyOne dataset to analyze.
            json_path (str): Path to the JSON file containing label remapping.
    """
    parser = argparse.ArgumentParser(description="Compare dataset similarity.")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="Name of the FiftyOne dataset to analyze.")
    
    parser.add_argument('--ratios',
                        type=float,
                        nargs=3,
                        default=[0.7, 0.15, 0.15],
                        help="""
                                List of ratios to compare dataset splits. For example, '0.7 0.15 0.15' will compare the first 70%, 15%, and 15% of the dataset.
                                (train, val, test). Default is [0.7, 0.15, 0.15].
                            """)
    
    parser.add_argument("--new_dataset_name",
                        type=str,
                        default=None,
                        help="new dataset name that you want to do all changes on it. If not provided, changes will be applied to the original dataset.")
    
    parser.add_argument('--untag_other',
                        action='store_true',
                        help="If set, removes all tags from samples except 'train', 'val', or 'test'.")
        
    parser.add_argument('--exclude_sample_ids',
                        type=str,
                        nargs='*',
                        default=[],
                        help="""
                                List of sample IDs to exclude from the train/val/test splitting.
                                These samples will be tagged as 'excluded'. For example, 'id1 id2 id3'.
                            """)
    
    parser.add_argument('--preserve_order_for_video_frames',
                        action='store_true',
                        help="""
                                If set, the non-excluded samples will NOT be shuffled before splitting.
                                This is crucial for datasets with video frames to prevent data leakage between splits.
                            """)
    
    parser.add_argument('--overwrite',
                        action='store_true',
                        help="If --new_dataset_name is provided and the dataset already exists, this flag allows overwriting it.")
                        
    args = parser.parse_args()
    return args


@parse_print_args
def run(ds_name, 
        ratios: list, 
        new_dataset_name=None, 
        untag_other=False, 
        exclude_sample_ids=None, 
        preserve_order_for_video_frames=False, 
        overwrite=False):
    """
    Splits a FiftyOne dataset into train/val/test by ratio and optionally removes old tags
    or creates a new dataset for the operation, with an option to exclude specific samples.
    If 'preserve_order_for_video_frames' is true, non-excluded samples are not shuffled before splitting.
    The 'overwrite' flag controls overwriting behavior for new datasets.
    """

    print(f"""
        =======================================================================================================

                                        🔍 CHECK DATASET {ds_name} RATIOS

        =======================================================================================================
        """
    )

    # Load original dataset
    if not fo.dataset_exists(ds_name):
        raise ValueError(f"❌ Dataset '{ds_name}' not found.")
    original_dataset = fo.load_dataset(ds_name)
    total_samples = len(original_dataset) # Total samples in the original source dataset
    print(f"✅ Loaded dataset: {ds_name} ({total_samples} samples)\n")

    # Validate ratios
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"❌ Invalid ratios {ratios}. Must be 3 values summing to 1.0.")

    # Determine the dataset to work on
    working_ds_name = ds_name
    
    if new_dataset_name:
        working_ds_name = new_dataset_name
        if fo.dataset_exists(working_ds_name):
            if overwrite:
                print(f"⚠️ Overwriting existing dataset '{working_ds_name}' as --overwrite is enabled...")
                fo.delete_dataset(working_ds_name)
            else:
                raise ValueError(f"❌ Dataset '{working_ds_name}' already exists. Use --overwrite to replace it, or choose a different --new_dataset_name.")
        
        print(f"🚀 Cloning '{ds_name}' to new dataset '{working_ds_name}' (this copies all samples and fields)...")
        # THIS IS THE CRUCIAL CHANGE: Calling clone() on the dataset directly copies its samples
        working_dataset = original_dataset.clone(working_ds_name) 
        print(f"✅ Dataset cloned and loaded: '{working_ds_name}' ({len(working_dataset)} samples)")
        # If the original dataset had a fixed_version, the cloned one should also have it.
        # This is more for consistency than a functional requirement for this script.
        working_dataset.persistent = original_dataset.persistent 
    else:
        working_dataset = original_dataset
        print(f"⚠️ Applying changes directly to original dataset: '{ds_name}'")

    # Now, working_dataset is either the original dataset or a full clone of it.
    # Proceed to filter samples from this working_dataset.

    excluded_samples_from_working_ds = []
    non_excluded_samples_from_working_ds = []
    
    print("Pre-processing samples to separate excluded IDs from the working dataset...")
    # Iterate through the working_dataset to filter samples
    for sample in tqdm(working_dataset, desc="Processing samples", unit="sample"):
        if exclude_sample_ids and sample.id in exclude_sample_ids:
            excluded_samples_from_working_ds.append(sample)
        else:
            non_excluded_samples_from_working_ds.append(sample)

    if exclude_sample_ids:
        print(f"✅ {len(excluded_samples_from_working_ds)} samples identified for exclusion.")
        print(f"✅ {len(non_excluded_samples_from_working_ds)} samples remaining for splitting.")
    else:
        print("No samples explicitly excluded via --exclude_sample_ids.")


    # Print current tag distribution of the working dataset BEFORE applying new splits
    all_tag_counts = defaultdict(int)
    for tag in working_dataset.distinct("tags"):
        count = len(working_dataset.match_tags(tag))
        if count > 0:
            all_tag_counts[tag] = count

    print("\n📊 Existing Split Tags (on working dataset):")
    if all_tag_counts:
        for tag, count in all_tag_counts.items():
            # Use total_samples for percentage relative to the original dataset size
            # (assuming total_samples from original_dataset is still relevant for percentages)
            print(f"{tag}: {count} samples ({count / total_samples:.2%})")
    else:
        print("No existing tags found on the working dataset.")


    # Print class distribution BEFORE splitting (on the entire original dataset for reference)
    print("\n🔍 Class distribution BEFORE re-splitting (on original dataset for reference):")
    for tag in ["train", "val", "test"]:
        tag_samples = original_dataset.match_tags(tag) # Still using original_dataset for "BEFORE" reference
        class_counter = Counter()
        for sample in tag_samples:
            if sample.get_field("ground_truth"):
                for det in sample.ground_truth.detections:
                    class_counter[det.label] += 1
        if class_counter:
            print(f"   📁 Split: {tag}")
            for cls, cnt in sorted(class_counter.items()):
                print(f"     🏷️ Class '{cls}': {cnt} instances")
        else:
             print(f"   📁 Split: {tag} (No samples with 'ground_truth' detections)")


    # Compute split sizes for non-excluded samples
    total_non_excluded = len(non_excluded_samples_from_working_ds)
    n_train = int(ratios[0] * total_non_excluded)
    n_val = int(ratios[1] * total_non_excluded)
    n_test = total_non_excluded - n_train - n_val
    print("\n📐 Target Split Sizes (for non-excluded samples):")
    print(f"train: {n_train}, val: {n_val}, test: {n_test}\n")

    # Conditional Shuffling
    if not preserve_order_for_video_frames:
        print("🔀 Shuffling non-excluded samples before splitting.")
        np.random.shuffle(non_excluded_samples_from_working_ds) # Shuffle in-place
    else:
        print("Maintaining original order for non-excluded samples (no shuffling) due to --preserve_order_for_video_frames.")
        # `non_excluded_samples_from_working_ds` is already in the iteration order of the dataset.

    split_samples = {
        "train": non_excluded_samples_from_working_ds[:n_train],
        "val": non_excluded_samples_from_working_ds[n_train:n_train + n_val],
        "test": non_excluded_samples_from_working_ds[n_train + n_val:]
    }

    print(f"⚙️ Applying new split tags to samples in '{working_ds_name}'...")

    # Reassign tags and save changes
    for split_name, samples in split_samples.items():
        print(f"🔖 Tagging {len(samples)} samples for '{split_name}'")
        for sample in tqdm(samples, desc=split_name, unit="sample"):
            if untag_other:
                sample.tags = [split_name]
            else:
                # Remove existing train/val/test/excluded tags, then add the new one
                sample.tags = list(set(sample.tags) - {"train", "val", "test", "excluded"}) + [split_name]
            sample.save() # Save changes to the sample in the database

    # Add "excluded" tag to excluded samples
    if excluded_samples_from_working_ds:
        print(f"🚫 Tagging {len(excluded_samples_from_working_ds)} samples as 'excluded'")
        for sample in tqdm(excluded_samples_from_working_ds, desc="Excluded samples", unit="sample"):
            if untag_other:
                sample.tags = ["excluded"]
            else:
                sample.tags = list(set(sample.tags) - {"train", "val", "test"}) + ["excluded"]
            sample.save() # Save changes to the sample in the database

    # Final tag distribution on the working dataset
    print(f"\n✅ Final Tag Distribution in '{working_ds_name}':")
    for tag in ["train", "val", "test", "excluded"]:
        count = len(working_dataset.match_tags(tag))
        if count > 0: # Only print if tag exists
            print(f"{tag}: {count} samples ({count / total_samples:.2%})") # total_samples refers to original total
        else:
            print(f"{tag}: 0 samples (0.00%)")


    # Class distribution AFTER re-splitting (considering only train/val/test for model training)
    print("\n📊 Class distribution AFTER re-splitting (for train/val/test splits):")
    for tag in ["train", "val", "test"]:
        tag_samples = working_dataset.match_tags(tag)
        class_counter = Counter()
        for sample in tag_samples:
            if sample.get_field("ground_truth"):
                for det in sample.ground_truth.detections:
                    class_counter[det.label] += 1
        if class_counter:
            print(f"   📁 Split: [green]{tag}")
            for cls, cnt in sorted(class_counter.items()):
                print(f"     🏷️ Class '{cls}': [blue]{cnt} instances")
        else:
            print(f"   📁 Split: {tag} (No samples with 'ground_truth' detections)")


    print(f"\n🎉 Done! Dataset '{working_ds_name}' has been updated.")


if __name__ == "__main__":
    args = init()
    run(args.ds_name, args.ratios, args.new_dataset_name, args.untag_other, args.exclude_sample_ids, args.preserve_order_for_video_frames, args.overwrite)