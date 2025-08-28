import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from rich import print

import fiftyone as fo

from shuttrix.tasks.utils import parse_print_args


def init():
    parser = argparse.ArgumentParser(description="Merge FiftyOne datasets.")
    parser.add_argument('--in_ds_names',
                        type=str,
                        nargs='+', 
                        required=True,
                        help="A list of dataset names to merge. E.g., 'ds1 ds2 ds3'")
    
    parser.add_argument('--merged_name',
                        type=str,
                        default="merged_dataset",
                        help="Name of the merged dataset (default: 'merged_dataset')")
    
    parser.add_argument('--overwrite',
                        action='store_true',
                        help="If set, an existing merged dataset with the same name will be deleted before merging.")
    
    parser.add_argument('--keep_only_split_tags',
                        action='store_true',
                        help="If set, only 'train', 'val', and 'test' tags will be preserved on samples in the merged dataset. All other tags will be removed.")

    args = parser.parse_args()
    return args


@parse_print_args
def run(in_ds_names: list, 
        merged_name: str, 
        overwrite: bool, 
        keep_only_split_tags: bool,
        ):
    """
        Merges a list of datasets provided as arguments into a single FiftyOne dataset,
        including only samples with 'train', 'test', or 'val' tags.
        Allows overwriting an existing merged dataset and controlling which tags are kept.
        
        Args:
            ds_name (list): A list of dataset names to be merged.
            merged_name (str): The name for the new merged dataset.
            overwrite (bool): If True, deletes an existing dataset with the same name.
            keep_only_split_tags (bool): If True, removes all tags except 'train', 'val', or 'test'.
            
    """
    print(f"📖 [green]Merging the following datasets: [blue]{in_ds_names}")

    # Filter out non-existing datasets and load them
    valid_datasets = []
    print("\n🔍 Checking source datasets...")
    for name in in_ds_names:
        if fo.dataset_exists(name):
            valid_datasets.append(fo.load_dataset(name))
            print(f"✅ Found dataset: [blue]'{name}'")
        else:
            print(f"⚠️ Warning: Dataset '{name}' does not exist and will be skipped.")
    
    if not valid_datasets:
        print("❌ No valid datasets found to merge after filtering.")
        return

    # Handle existing merged dataset based on overwrite flag
    if fo.dataset_exists(merged_name):
        if overwrite:
            print(f"⚠️ Overwriting existing dataset: {merged_name}")
            fo.delete_dataset(merged_name)
        else:
            raise ValueError(f"❌ Merged dataset '{merged_name}' already exists. Use --overwrite to replace it.")

    merged_dataset = fo.Dataset(name=merged_name)

    # Define common split tags
    split_tags = {"train", "test", "val"}

    # Count total valid samples that will be merged
    total_samples_to_merge = 0
    for dataset in valid_datasets:
        for sample in dataset:
            if sample.tags and any(t in sample.tags for t in split_tags):
                total_samples_to_merge += 1

    print(f"\n🔄 Merging {len(valid_datasets)} datasets ({total_samples_to_merge} samples expected to be added) into '{merged_name}'")
    if keep_only_split_tags:
        print("📝 Only 'train', 'val', 'test' tags will be kept on merged samples.")
    else:
        print("📝 All original tags will be preserved on merged samples.")


    with tqdm(total=total_samples_to_merge, desc="Merging samples", unit="samples") as pbar:
        for dataset in valid_datasets:
            for sample in dataset:
                current_sample_tags = set(sample.tags or []) # Ensure it's a set for efficient checking

                # Only include samples that have at least one of the defined split_tags
                if current_sample_tags and any(t in current_sample_tags for t in split_tags):
                    # Handle tags based on the new argument
                    if keep_only_split_tags:
                        sample.tags = list(current_sample_tags & split_tags) # Intersection
                    # If not keep_only_split_tags, sample.tags remains as is (current_sample_tags)
                    
                    merged_dataset.add_sample(sample)
                    pbar.update(1)

    print(f"\n✅ Successfully merged {len(merged_dataset)} samples into '{merged_name}'")

    # Final verification/summary (optional, but good practice)
    print("\n📊 Final Tag Distribution in Merged Dataset:")
    merged_tag_counts = defaultdict(int)
    for tag in merged_dataset.distinct("tags"):
        count = len(merged_dataset.match_tags(tag))
        if count > 0:
            merged_tag_counts[tag] = count
    
    if merged_tag_counts:
        for tag, count in sorted(merged_tag_counts.items()):
            print(f"  '{tag}': {count} samples ({count / len(merged_dataset):.2%})")
    else:
        print("No tags found in the merged dataset.")


if __name__ == "__main__":
    args = init()
    run(args.ds_name, 
        args.merged_name, 
        args.overwrite, 
        args.keep_only_split_tags)