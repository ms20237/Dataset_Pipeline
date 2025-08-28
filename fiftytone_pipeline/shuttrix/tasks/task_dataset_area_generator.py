import argparse 
from rich import print
from tqdm import tqdm

import fiftyone as fo
from fiftyone import ViewField as F

from shuttrix.tasks.utils import parse_print_args


def init():
    parser = argparse.ArgumentParser(description="Generate dataset area from FiftyOne dataset")
    parser.add_argument("--ds_name",
                        type=str,
                        required=True,
                        help="Name of the FiftyOne dataset to process",
    )
    
    return parser.parse_args()


@parse_print_args
def run(ds_name: str):
    print(f"""
            =======================================================================================================
            
                                                GENERATE AREA FIELD IN {ds_name}
                                            
            =======================================================================================================
            """ 
    )
    
    if not fo.dataset_exists(ds_name):
        print(f"❌ Dataset '{ds_name}' not found.")
        return

    dataset = fo.load_dataset(ds_name)
    print(f"✅ Loaded dataset '{ds_name}' with {len(dataset)} samples")

    # Add area fields to schema if they don't exist
    if "ground_truth.detections.area" not in dataset.get_field_schema():
        dataset.add_sample_field("ground_truth.detections.area", fo.FloatField)
    if "ground_truth.area" not in dataset.get_field_schema():
        dataset.add_sample_field("ground_truth.area", fo.FloatField)
    if "area" not in dataset.get_field_schema():
        dataset.add_sample_field("area", fo.FloatField)

    for sample in tqdm(dataset, desc="Processing samples"):
        if "ground_truth" in sample and sample.ground_truth is not None:
            detections = sample.ground_truth.detections
            img_width = sample.metadata.width
            img_height = sample.metadata.height

            total_area = 0.0
            for det in detections:
                x, y, w, h = det.bounding_box
                abs_area = w * img_width * h * img_height
                det.area = abs_area  # set detection-level area
                total_area += abs_area

            # update detection list
            sample.ground_truth.detections = detections
            # set ground_truth.area (sum of detection areas)
            sample.ground_truth.area = total_area
            # set sample-level area (optional, for global filtering)
            sample.area = total_area
            sample.save()

    print("✅ Done: area field added to each sample for filtering.")
    

if __name__ == "__main__":
    args = init()
    run(args.ds_name)
    print(f"Dataset area generation completed for {args.ds_name} using label field 'area'.")    
    