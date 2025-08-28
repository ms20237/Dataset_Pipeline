import argparse 
from tqdm import tqdm
from rich import print

import fiftyone as fo

from shuttrix.tasks.utils import parse_print_args


def init():
    parser = argparse.ArgumentParser(description="Generate bbox values field")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="dataset name exp:ds1, ds2, ...")
    
    args = parser.parse_args()
    return args


@parse_print_args
def run(ds_name: str):
    """
        Adds normalized and absolute bounding box width and height fields to each detection in the 'ground_truth' field of a FiftyOne dataset.
        Args:
            ds_name (str): The name of the FiftyOne dataset to process.
        This function:
            - Loads the specified FiftyOne dataset.
            - Adds new fields to the detection schema for normalized and absolute bounding box dimensions.
            - Iterates over each sample and detection, calculating and assigning the new fields.
            - Saves the updated samples back to the dataset.
        The new fields added to each detection are:
            - bbox_norm_width: Normalized width of the bounding box (relative to image width).
            - bbox_norm_height: Normalized height of the bounding box (relative to image height).
            - bbox_abs_width: Absolute width of the bounding box in pixels.
            - bbox_abs_height: Absolute height of the bounding box in pixels.
    """
    print(f"""
            =======================================================================================================
            
                                                GENERATE BBOX FIELD IN {ds_name}
                                            
            =======================================================================================================
            """ 
    )
    # load dataset
    dataset = fo.load_dataset(ds_name)
    print(f"✅ Loaded dataset '{ds_name}' with {len(dataset)} samples")

    # Add detection fields to schema
    dataset.add_sample_field("ground_truth.detections.bbox_norm_width", fo.FloatField)
    dataset.add_sample_field("ground_truth.detections.bbox_norm_height", fo.FloatField)
    dataset.add_sample_field("ground_truth.detections.bbox_abs_width", fo.IntField)
    dataset.add_sample_field("ground_truth.detections.bbox_abs_height", fo.IntField)

    labels = dataset.distinct("ground_truth.detections.label")
    print("Labels in dataset: ", labels)

    for sample in tqdm(dataset, desc="Updating samples"):
        width = sample.metadata.width
        height = sample.metadata.height
        dets = sample["ground_truth"].detections
        
        for det in dets:
            x, y, w, h = det.bounding_box
            det.bbox_norm_width = w
            det.bbox_norm_height = h
            det.bbox_abs_width = int(w * width)
            det.bbox_abs_height = int(h * height)
            
        sample["ground_truth"].detections = dets
        sample.save()
        
    print("✅ Done: bbox fields added to each sample for filtering.")
    
if __name__ == "__main__":
    args = init()
    run(args.ds_name)    
    

