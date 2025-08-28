import argparse
from tqdm import tqdm
from collections import Counter, defaultdict
from rich import print
from pprint import pprint

import fiftyone as fo

from shuttrix.tasks.utils import parse_print_args


def init():
    """
    Initializes and parses command-line arguments for uniformizing a FiftyOne dataset.
    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            ds_name (str): Name of the dataset (e.g., ds1, ds2, ...).
            output_name (str, optional): The name for the output uniform dataset.
    """
    parser = argparse.ArgumentParser(description="Uniformize FiftyOne dataset labels.")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="dataset name exp:ds1, ds2, ...")
    
    parser.add_argument('--output_name',
                        type=str,
                        required=False,
                        help="Name of the output uniform dataset. If not provided, the original dataset will be overwritten.")
    
    args = parser.parse_args()
    return args
    
    
@parse_print_args
def run(ds_name: str, 
        output_name: str,
    ):
    """
    Uniformizes the number of samples per label in a FiftyOne dataset for each tag ('train', 'val', 'test').
    This function loads a FiftyOne dataset by name, analyzes the distribution of labels in the 'ground_truth'
    field, and creates a new dataset where each label has the same number of samples within each tag. The
    minimum sample count across all labels for each tag is used as the uniform count. The new uniform dataset
    is saved with the specified output name, overwriting the original if no output name is provided.
    
    Args:
        ds_name (str): The name of the existing FiftyOne dataset to process.
        output_name (str): The name for the new uniform dataset.
    
    Behavior:
        - Loads the specified dataset or creates a new one if it does not exist.
        - Prints the available labels and their sample counts.
        - Identifies the tags present in the dataset (among 'train', 'val', 'test').
        - For each tag, selects an equal number of samples for each label (the minimum count across labels).
        - Creates a new dataset containing the uniform samples for each tag.
        - Prints summary statistics throughout the process.
    
    Notes:
        - Only samples with non-empty 'ground_truth' detections are considered.
        - If a uniform dataset with the target name already exists, it is deleted before creation.
        - The function prints informative messages and statistics at each step.
    """
    print(f"""
        =======================================================================================================
            
                                         UNIFORM NUMBER OF EACH LABELS in {ds_name}
            
        =======================================================================================================
    """
    )
        
    # load dataset
    if fo.dataset_exists(ds_name):
        dataset = fo.load_dataset(ds_name)
        print(f"✅ Loaded dataset: {ds_name} ({len(dataset)} samples)")
    else:
        print(f"❌ Dataset '{ds_name}' not found. Exiting.")
        return
        
    # print labels in dataset
    labels = dataset.distinct("ground_truth.detections.label")
    print("📋 Labels in dataset:")
    pprint(labels)
    
    # Count how many samples each label appears in
    label_sample_counts = defaultdict(int)

    for sample in dataset:
        if "ground_truth" not in sample or sample["ground_truth"] is None:
            continue

        detections = sample["ground_truth"].detections
        sample_labels = set(det.label for det in detections)

        for label in sample_labels:
            label_sample_counts[label] += 1

    # Print counts
    if label_sample_counts:
        print("\n📊 Sample count per label:")
        for label, count in label_sample_counts.items():
            print(f"  {label}: {count} samples")

        # Get label with minimum samples
        min_label, min_count = min(label_sample_counts.items(), key=lambda x: x[1])
        print(f"\n🔍 Label with fewest samples: '{min_label}' ({min_count} samples)")
    else:
        print("❌ No labeled detections found in any samples.")
        
        
    # ------ Uniform the dataset to have an equal number of samples for each label in each tag[train, val, test] -------
    # Detect which tags (train, test, val) are present
    all_tags = set(tag for sample in dataset for tag in sample.tags)
    valid_tags = [t for t in ["train", "test", "val"] if t in all_tags]
    
    if not valid_tags:
        print("❌ No valid tags (train, test, val) found in dataset.")
        return
    
    print(f"📌 Found tags in dataset: [green]{sorted(all_tags)}")
    print(f"📂 Will process these tags: [green]{valid_tags}")
        
    # Use the provided output_name, or default to the original dataset name
    uniform_name = output_name if output_name else ds_name

    # List to hold the samples that will be added to the uniform dataset
    uniform_samples_list = []
    
    print(f"\n🔄 Collecting uniform samples for dataset '{uniform_name}'")

    # Iterate over each tag and collect uniform sample IDs
    for tag in valid_tags:
        print(f"\n🔎 Processing tag: {tag}")
        tag_view = dataset.match_tags(tag)

        # Build label -> list of sample IDs for this tag
        label_to_sample_ids = defaultdict(list)
        for sample in tag_view:
            if "ground_truth" not in sample or sample["ground_truth"] is None:
                continue
            
            detections = sample["ground_truth"].detections
            labels_in_sample = set(det.label for det in detections)
            
            for label in labels_in_sample:
                label_to_sample_ids[label].append(sample.id)

        if not label_to_sample_ids:
            print(f"⚠️ No labels found in tag '{tag}'. Skipping.")
            continue

        # Find minimum number of samples per label     
        min_count = min(len(sample_ids) for sample_ids in label_to_sample_ids.values())
        print(f"✅ Minimum count per label in '{tag}': {min_count}")
        
        selected_ids_for_tag = set()
        
        # Take a uniform number of sample IDs for each label
        for label, sample_ids in label_to_sample_ids.items():
            selected_ids_for_tag.update(sample_ids[:min_count])

        print(f"📦 Collected {len(selected_ids_for_tag)} unique sample IDs for tag '{tag}'")
        
        # Load the samples by ID and add them to the list
        samples_from_ids = list(dataset.select(list(selected_ids_for_tag)))

        for sample in samples_from_ids:
            sample.tags = [tag]
        uniform_samples_list.extend(samples_from_ids)
        
    print(f"\n🔄 Total samples collected across all tags: {len(uniform_samples_list)}")
    
    # Now that all samples are collected, create or overwrite the final dataset
    if fo.dataset_exists(uniform_name):
        print(f"⚠️ Deleting existing dataset: {uniform_name}")
        fo.delete_dataset(uniform_name)
        
    uniform_ds = fo.Dataset(name=uniform_name)
    uniform_ds.add_samples(uniform_samples_list)

    print(f"\n✅ Successfully created uniform dataset '{uniform_name}' with {len(uniform_ds)} samples")
    
    # Final verification/summary
    print("\n📊 Final Tag Distribution in Uniform Dataset:")
    merged_tag_counts = defaultdict(int)
    for tag in uniform_ds.distinct("tags"):
        count = len(uniform_ds.match_tags(tag))
        if count > 0:
            merged_tag_counts[tag] = count
    
    if merged_tag_counts:
        for tag, count in sorted(merged_tag_counts.items()):
            print(f"  '{tag}': {count} samples ({count / len(uniform_ds):.2%})")
    else:
        print("No tags found in the merged dataset.")


if __name__ == "__main__":
    args = init()
    run(args.ds_name, 
        args.output_name)
