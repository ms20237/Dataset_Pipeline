import argparse
import json
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
            priority_json_path (str, optional): Path to a JSON file for sample priority.
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
    
    parser.add_argument('--priority_json_path',
                        type=str,
                        required=False,
                        help="Path to a JSON file specifying priority for datasets per label.")
    
    args = parser.parse_args()
    return args

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

@parse_print_args
def run(ds_name: str, 
        output_name: str,
        priority_json_path: str = None
    ):
    """
    Uniformizes the number of samples per label in a FiftyOne dataset for each tag ('train', 'val', 'test').
    
    Args:
        ds_name (str): The name of the existing FiftyOne dataset to process.
        output_name (str): The name for the new uniform dataset.
        priority_json_path (str, optional): Path to a JSON file for sample priority.
    """
    print(f"""
        =======================================================================================================
          
                              UNIFORM NUMBER OF EACH LABELS in {ds_name}
          
        =======================================================================================================
    """
    )
        
    # Load dataset
    if fo.dataset_exists(ds_name):
        dataset = fo.load_dataset(ds_name)
        print(f"✅ Loaded dataset: {ds_name} ({len(dataset)} samples)")
    else:
        print(f"❌ Dataset '{ds_name}' not found. Exiting.")
        return
        
    # Load priority JSON if path is provided
    priority_config = {}
    if priority_json_path:
        try:
            with open(priority_json_path, 'r') as f:
                priority_config = json.load(f)
            print(f"✅ Loaded priority configuration from '{priority_json_path}'")
        except FileNotFoundError:
            print(f"⚠️ Priority JSON file not found at '{priority_json_path}'. Continuing without prioritization.")
        except json.JSONDecodeError:
            print(f"⚠️ Invalid JSON format in '{priority_json_path}'. Continuing without prioritization.")
    
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

        # Build label -> list of samples for this tag
        label_to_samples = defaultdict(list)
        for sample in tag_view.select_fields(["tags", "ground_truth"]):
            if "ground_truth" not in sample or sample["ground_truth"] is None:
                continue
                
            detections = sample["ground_truth"].detections
            labels_in_sample = set(det.label for det in detections)
            
            # Use the sample's tags to determine its source dataset
            # Samples can have multiple tags, so we need to handle that.
            source_tags = sample.tags
            
            for label in labels_in_sample:
                label_to_samples[label].append((sample.id, source_tags))

        if not label_to_samples:
            print(f"⚠️ No labels found in tag '{tag}'. Skipping.")
            continue

        # Find minimum number of samples per label
        min_count = min(len(samples) for samples in label_to_samples.values())
        print(f"✅ Minimum count per label in '{tag}': {min_count}")
        
        selected_ids_for_tag = set()
        
        # Take a uniform number of sample IDs for each label, with priority
        for label, samples_with_sources in label_to_samples.items():
            
            # Sort samples based on priority if the label exists in the config
            if label in priority_config:
                priority_list = priority_config[label]
                
                # Sort the list of (id, source_tags) tuples based on the priority of the source tag
                samples_with_sources.sort(key=lambda x: priority_list.index(get_source_tag(dataset[x[0]], priority_list)) if get_source_tag(dataset[x[0]], priority_list) in priority_list else len(priority_list))
                print(f"✨ Prioritizing samples for label '{label}' based on JSON config.")
            
            # Take the top `min_count` samples
            selected_ids = [sample_id for sample_id, _ in samples_with_sources[:min_count]]
            selected_ids_for_tag.update(selected_ids)

        print(f"📦 Collected {len(selected_ids_for_tag)} unique sample IDs for tag '{tag}'")
        
        # Load the samples by ID and add them to the list
        samples_from_ids = list(dataset.select(list(selected_ids_for_tag)))

        for sample in samples_from_ids:
            # We don't want to overwrite the original dataset tags, so we'll just set the split tag
            split_tag = [t for t in sample.tags if t in ["train", "val", "test"]]
            sample.tags = split_tag
            sample.tags.append(tag)
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
        args.output_name,
        args.priority_json_path)
    
    