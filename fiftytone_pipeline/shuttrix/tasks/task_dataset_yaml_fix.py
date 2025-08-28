import os
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from rich import print

from shuttrix.tasks.utils import parse_print_args, find_split


def init():
    """
        
    
    """
    parser = argparse.ArgumentParser(description="Fix yaml files of dataset.")
    parser.add_argument('--datasets_dir',
                        type=str,
                        required=True,
                        help="Path to the directory containing the dataset archives.")
 
    args = parser.parse_args()
    return args


@parse_print_args
def run(datasets_dir):
    print(
        """
            =======================================================================================================

                                                FIX YAML FILES OF DATASETS

            =======================================================================================================
        """
    )

    datasets_dir = Path(datasets_dir)
    if not datasets_dir.exists():
        print(f"❌ Directory not found: {datasets_dir}")
        return

    yaml_files = list(datasets_dir.rglob("*.yaml"))

    if not yaml_files:
        print("⚠️ No YAML files found in the entire directory tree.")
        return

    for yaml_path in tqdm(yaml_files, desc="🗂️ Processing YAML files"):
        dataset_dir = yaml_path.parent
        original_name = yaml_path.name
        target_yaml_name = "dataset.yaml"
        new_yaml_path = dataset_dir / target_yaml_name

        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"❌ Failed to read {original_name}: {e}")
            continue
        
        # Recursive search for 'images/train', 'images/val', etc.
        train_dir = find_split("train", dataset_dir)
        val_dir = find_split("val", dataset_dir)
        test_dir = find_split("test", dataset_dir)
        
        if train_dir is not None:
            data["train"] = str(train_dir.resolve())
        elif "train" in data:
            print("⚠️ Could not find train split, keeping old value")

        if val_dir is not None:
            data["val"] = str(val_dir.resolve())
        elif "val" in data:
            print("⚠️ Could not find val split, keeping old value")

        if test_dir is not None:
            data["test"] = str(test_dir.resolve())
        elif "test" in data:
            print("⚠️ Could not find test split, keeping old value")


        print(f"📄 Processing: {yaml_path}")
        print("🔍 Updated splits:")
        print(f"  • train: {data.get('train', '❌ Not found')}")
        print(f"  • val:   {data.get('val', '❌ Not found')}")
        print(f"  • test:  {data.get('test', '❌ Not found')}")

        if original_name != target_yaml_name:
            try:
                yaml_path.unlink()
                print(f"🔄 Renamed '{original_name}' → '{target_yaml_name}'")
            except Exception as e:
                print(f"⚠️ Could not delete old YAML: {e}")

        try:
            with open(new_yaml_path, "w") as f:
                yaml.dump(data, f, sort_keys=False)
            print(f"✅ Fixed and saved: {new_yaml_path}")
        except Exception as e:
            print(f"❌ Failed to write YAML: {e}")
        

if __name__ == "__main__":
    args = init()
    run(args.datasets_dir)
    
        