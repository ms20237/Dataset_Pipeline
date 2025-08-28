import os
import argparse
from rich import print

import fiftyone as fo
import fiftyone.types as fot

from shuttrix.tasks.utils import parse_print_args


def init():
    parser = argparse.ArgumentParser(description="Export FiftyOne dataset.")
    parser.add_argument('--ds_name', 
                        type=str, 
                        required=True,
                        help="dataset name exp: ds1, ds2, ...")
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        required=True,
                        help="output directory to save the exported dataset")
    
    parser.add_argument('--format', 
                        type=str, 
                        default="yolov5",
                        help="export format, e.g., 'yolo', 'coco', 'fiftyone', etc.")
    
    parser.add_argument('--label_list', 
                        type=str, 
                        required=True,
                        help="Comma-separated list of class labels to include in the export (e.g., 'cat,dog,car')")
    
    parser.add_argument('--splits', 
                        type=str, 
                        default=None,
                        help="Comma-separated list of splits to export (e.g., 'train,val'). If None, all splits are exported.")
    
    return parser.parse_args()


@parse_print_args
def run(ds_name: str,
        output_dir: str,
        format: str,
        label_list: str,
        splits: str):

    print(f"""
        =======================================================================================================
        
                                         EXPORTING {ds_name} FROM FIFTYONE
                                         
        =======================================================================================================
    """)

    if not fo.dataset_exists(ds_name):
        print(f"❌ Dataset '{ds_name}' not found.")
        return

    dataset = fo.load_dataset(ds_name)
    print(f"✅ Loaded dataset '{ds_name}' with {len(dataset)} samples")

    classes = [label.strip() for label in label_list.split(',')] if label_list else None
    if classes:
        print("🔢 Label priority (class IDs):")
        for i, label in enumerate(classes):
            print(f"  {i}: {label}")

    export_format_map = {
        "yolov5": fot.YOLOv5Dataset,
        "yolov8": fot.YOLOv5Dataset,
        "coco": fot.COCODetectionDataset,
        "voc": fot.VOCDetectionDataset,
        "kitti": fot.KITTIDetectionDataset,
        "fiftyone_image_classification": fot.FiftyOneImageClassificationDataset,
    }

    export_type = export_format_map.get(format)
    if not export_type:
        print(f"❌ Export failed: Unsupported export format '{format}'.")
        return

    # Determine splits
    if splits:
        split_list = [s.strip() for s in splits.split(',')]
    else:
        # Export all splits found in tags
        all_tags = set()
        for sample in dataset:
            all_tags.update(sample.tags)
        split_list = list(all_tags)
        print(f"📑 Auto-detected splits from tags: {split_list}")

    for split in split_list:
        view = dataset.match_tags(split)
        if len(view) == 0:
            print(f"⚠️  No samples found for split '{split}', skipping...")
            continue

        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        try:
            view.export(
                export_dir=split_output_dir,
                dataset_type=export_type,
                label_field="ground_truth",
                classes=classes,
                overwrite=True
            )
            print(f"✅ Exported split '{split}' to '{split_output_dir}'")
        except Exception as e:
            print(f"❌ Failed to export split '{split}': {e}")


if __name__ == "__main__":
    args = init()
    run(args.ds_name,
        args.output_dir,
        args.format,
        args.label_list,
        args.splits)
